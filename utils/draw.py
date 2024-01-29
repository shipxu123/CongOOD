import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def draw_cong():
    print(os.listdir("./collected"))
    n = len(os.listdir("./collected"))
    count = 0
    for file in os.listdir("./collected"):
        file_path = "./collected/" + file
        with open(file_path, "rb") as f:
            print("[Drawing] " + file[:-4] + f"------[{count+1}/{n}]")
            data = pickle.load(f)
            congV = data['congV']
            congH = data['congH']
            macro = data['macro']
            density = data['density']
            rudy = data['rudy']

            draw_congV = np.zeros((congV.shape[0],congV.shape[1],3))
            draw_congV[:,:,0] = congV

            draw_congH = np.zeros((congH.shape[0],congH.shape[1],3))
            draw_congH[:,:,1] = congH

            draw_macro = np.zeros((512,512,3))
            draw_macro[:,:,2] = macro

            draw_density = np.zeros((512,512,3))
            draw_density[:,:,0] = density
            draw_density[:,:,1] = density

            draw_rudy = np.zeros((512,512,3))
            draw_rudy[:,:,1] = rudy
            draw_rudy[:,:,2] = rudy

            plt.figure(figsize=(11,3))
            plt.subplot(151)
            plt.title("CongV")
            plt.imshow(draw_congV, interpolation='nearest')

            plt.subplot(152)
            plt.title("CongH")
            plt.imshow(draw_congH, interpolation='nearest')

            plt.subplot(153)
            plt.title("macro")
            plt.imshow(draw_macro, interpolation='nearest')

            plt.subplot(154)
            plt.title("density")
            plt.imshow(draw_density, interpolation='nearest')

            plt.subplot(155)
            plt.title("rudy")
            plt.imshow(draw_rudy, interpolation='nearest')
            save_path = "./fig/" + file[:-4]+".png"
            plt.savefig(save_path)
            count += 1
            plt.close()