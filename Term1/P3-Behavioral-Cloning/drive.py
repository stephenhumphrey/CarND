import argparse
import base64
from datetime import datetime
import time
import os
import sys
import select
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from collections import defaultdict

doWithoutKeras = False # skip using Keras, initially

if not doWithoutKeras:
    from keras.models import load_model
    import h5py
    from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# **************** Joystick stuff ****************

doDescriptiveLabels = True # display descriptive names instead of just magic numbers
doPruneEmptyStatus = True # if True, remove 0 values from the button/axis status dictionary

def processJoystickEvent(buffer):
    # get the event type, and either the button or axis, depending on the event type
    items = np.frombuffer(buffer, dtype=np.uint8, count=2, offset=6)
    event = items[0]
    buttonOrAxis = items[1]

    # get the value of the button or joystick axis
    value = np.frombuffer(buffer, dtype=np.int16, count=1, offset=4)[0]

    # get the time in milliseconds (since when?) of the event
    time = np.frombuffer(buffer, dtype=np.uint32, count=1, offset=0)[0]

    return ( int(event), int(buttonOrAxis), int(value), int(time) )

def descriptiveJoystickLabels():
    """ descriptive versions of event and device numbers """

    eventType = 'eventType'
    deviceType = 'deviceType'
    eventButtonChanged = { eventType: "button-changed", deviceType: "button" }
    eventAxisMoved = { eventType: "axis-moved", deviceType: "axis" }
    eventInitButton = { eventType: "initial-value", deviceType: "button" }
    eventInitAxis = { eventType: "initial-axis", deviceType: "axis" }
    eventUnknown = { eventType: "unknown-event", deviceType: "device" }
    joystickEvents = defaultdict( lambda: eventUnknown )
    joystickEvents[1] = eventButtonChanged
    joystickEvents[2] = eventAxisMoved
    joystickEvents[129] = eventInitButton
    joystickEvents[130] = eventInitAxis

    return ( joystickEvents, eventType, deviceType )

def captureJoystickEvents( joystick = 0, maximumEvents = 0, status = None ):
    """ threadable Joystick polling process """

    if doDescriptiveLabels:
        joystickEvents, eventType, deviceType = descriptiveJoystickLabels()

    with open('/dev/input/js{}'.format(joystick), 'rb') as js:
        dataFrameSize, dataFrameCursor = 8, 0
        buffer = np.zeros( shape=(dataFrameSize,), dtype=np.uint8 )

        eventsBeforeQuitting = maximumEvents
        while eventsBeforeQuitting > 0 or maximumEvents == 0:

            buffer[ dataFrameCursor ] = np.uint8( ord( js.read(1) ) )
            dataFrameCursor += 1

            if dataFrameCursor >= dataFrameSize:
                dataFrameCursor = 0

                event, axis, value, time = processJoystickEvent( buffer[:dataFrameSize] )

                if doDescriptiveLabels:
                    type, device = ( joystickEvents[event][eventType],
                                     joystickEvents[event][deviceType] )
                else:
                    type, device = event, "device{}-".format( event )

                msg = "Joystick {} event [{}] on {}{}: value = {} at time = {}\n"
                sys.stdout.write( msg.format( joystick, type, device, axis, value, time ) )
                sys.stdout.flush()

                if status is not None:
                    key = "js{}{}{}".format( joystick, device, axis)
                    status[ key ] = value

                    if doPruneEmptyStatus and status[ key ] == 0:
                        del status[ key ]

                eventsBeforeQuitting -= 1

                if False and ( event, axis, value ) == ( 1, 0, 1 ): # "pulled the trigger!"
                    break;
    return

if True:
    from concurrent.futures import ThreadPoolExecutor

    joystickStatus = defaultdict(int)

    executor = ThreadPoolExecutor( max_workers = 2 )
    joystickThread0 = executor.submit( captureJoystickEvents, joystick = 0, status = joystickStatus )
    joystickThread1 = executor.submit( captureJoystickEvents, joystick = 1, status = joystickStatus )

# **************** speed and error controllers ****************

class SimplePIDcontroller:
    def __init__( self, Kp, Ki, Kd = 0.0, initialIntegral = 0.0 ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = 0.
        self.error = 0.
        self.integral = initialIntegral
        self.priorTime = time.time()
        self.priorError = 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def get_desired(self):
        return float( self.set_point )

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        # derivative error
        delta = 0.0
        if self.Kd > 0:
            deltaError = self.error - self.priorError
            deltaTime = time.time() - self.priorTime
            if deltaTime > 0:
                delta = deltaError / deltaTime

        # prepare for future derivatives
        self.priorError = self.error
        self.priorTime = time.time()

        return self.Kp * self.error + self.Ki * self.integral + self.Kd * delta


errorController = SimplePIDcontroller( 0.45, 0.00030, 0.15 ) # 0.55, 0.0005, 0.10
errorController.set_desired( 0.00 )

speedController = SimplePIDcontroller( 0.1, 0.0060, 0.15 ) # 0.1, 0.008, 0.00
set_speed = 13.0 # 9.0 got around complex track, one weird reversal, let's try 11.0
speedController.set_desired(set_speed)

steeringAngles = [0,] * 9 # rolling average accumulator

@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        global steeringAngles

        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]

        print( "telemetry angle:", steering_angle, "throttle:", throttle, "speed:", speed )

        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        if doWithoutKeras:
            steering_angle = joystickStatus["js1axis0"] / 32768
            print( "override angle:", steering_angle )
        else:
            trigger = joystickStatus["js1button0"]
            if trigger > 0:
                # while the joystick trigger is down, control the run manually
                steering_angle = joystickStatus["js1axis0"] / 32768
                print( "override angle:", steering_angle )
            else:
                img_shape = (160,320,3)
                margin = 100  # reduce the total width of the image by this amount
                left = int(margin / 2)
                right = img_shape[1] - left
                topMargin = 55
                bottomMargin = 25

                inferredError = float(
                    model.predict(image_array[None, topMargin:-bottomMargin, left:right, :], batch_size=1))

                steering_angle = errorController.update( inferredError / (right - left) )
                steering_angle = max( min( steering_angle, 1.0 ), -1.0 )

                print( "inferred error:", inferredError, "angle:", steering_angle )


        if float( speed ) > ( set_speed * 1.5 ):

            # brake with a negative throttle
            throttle = float( -0.25 * ( ( float( speed ) / set_speed ) - 1.0 ) )
            throttle = min( max( throttle, -0.9 ), -0.0005 )

            brake = 1.00
        else:
            throttle = speedController.update(float(speed))
            throttle = max( min( throttle, 0.95 ), 0.0005 )
            brake = 0.00

        # calculate short-window rolling average of the inferred steering angles
        # this is the equivalent of giving a few prior frames a vote on the current angle,
        # so as to smooth out outliers in the inferences
        steeringAngles = steeringAngles[1:] + [steering_angle]
        steering_angle = sum(steeringAngles) / len(steeringAngles)

        print( "control angle:", steering_angle, "throttle:", throttle, "brake:", brake )

        send_control( steering_angle, throttle )

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    if doWithoutKeras: # skip Keras for now
        model = None
    else:
        f = h5py.File(args.model, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                  ', but the model was built using ', model_version)

        model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
