#!/usr/bin/env python

from http.server import BaseHTTPRequestHandler, HTTPServer
from utils import take_screenshot, prepare_image
from utils import XboxController
import tensorflow as tf
import model
from termcolor import cprint
import wx
from scipy import ndimage, misc
import pickle

import numpy as np
PORT_NUMBER = 8082

# Start session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Load Model
saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")

# Init contoller for manual override
real_controller = XboxController()

# Play
class myHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        ## Look
        bmp = take_screenshot()
        #image_file = 'sol/'+'img_tmp.png'
        #bmp.SaveFile(image_file, wx.BITMAP_TYPE_PNG)
        #image = ndimage.imread(image_file, mode="RGB")
        #img = bmp.ConvertToImage()
        vec = prepare_image(bmp)

        ## Think
        joystick = model.y.eval(feed_dict={model.x: [vec], model.keep_prob: 1.0})[0]
        joystick = np.append(joystick,[0])
        print(joystick)

        ## Act
        ### manual override
        manual_override = real_controller.manual_override()

        if (manual_override):
            joystick = real_controller.read()
            print(joystick)
            joystick[1] *= -1 # flip y (this is in the config when it runs normally)

        ### calibration
        output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
            int(round(joystick[5])),
        ]

        ### print to console
        if (manual_override):
            cprint("Manual: " + str(output), 'yellow')
        else:
            cprint("AI: " + str(output), 'green')

        ### respond with action
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        data_string = pickle.dumps(output)
        self.wfile.write(bytes(data_string))
        return


if __name__ == '__main__':
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    print ('Started httpserver on port ' , PORT_NUMBER)
    server.serve_forever()
