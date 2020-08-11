# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:20:51 2019

@author: Medhavi
"""


from PIL import ImageTk,Image
import tkinter as tk
import cv2,time
import pyopencl as cl
import numpy as np

root = tk.Tk()
root.geometry("800x400+200+100")
w = 200
h = 200
blocks = 3
shift = 10

c = [None for i in range(blocks)]
for i in range(blocks):
    c[i] = tk.Canvas(root, bd=2, bg="red", height=h, width=w)
    c[i].grid(column = i, row = 0)



# Read image 
inputimg = None
inputim = None
predimg = None
predim = None
expectimg = None
expectim = None
video = None
n=8


row = 75
col = 78

#weights =  np.genfromtxt("weightsGPU.csv", delimiter=',')
#bias = np.genfromtxt("biasGPU.csv", delimiter=',')

start = time.time()
weights =  np.random.rand(row*col,row*col)
start = time.time() - start
print("Weights generation time ",start)
bias = np.random.rand(row*col)

"""
start = time.time()
weights =  np.load("weightsGPU.npy")
start = time.time() - start
print("Weights load time ",start)

bias = np.load("biasGPU.npy")
"""

training_inputs = []
target_outputs = []

for i in range(1):
    arr1 = np.genfromtxt("m ("+str(i)+").csv", delimiter=',')
    print(arr1.shape)
    arr1 = arr1.flatten()
    training_inputs.append(arr1)
    print(arr1.shape,training_inputs[0].shape)
    arr2 = np.genfromtxt("t ("+str(i)+").csv", delimiter=',')
    arr2 = arr2.flatten()
    target_outputs.append(arr2)
    print(arr2.shape,target_outputs[0].shape)

#print(weights,bias)


training_inputs = np.array(training_inputs).astype(np.float32)
target_outputs = np.array(target_outputs).astype(np.float32)

platform = cl.get_platforms()[1]
#print(platform)
device = platform.get_devices()[0]
#print(device)
ctx = cl.Context([device])
#print(ctx)
queue = cl.CommandQueue(ctx)
#print(queue)
mf = cl.mem_flags
out = cl.Program(ctx, """
    __kernel void output(int roww,int colw, __global float *wei ,__global float *bi ,
                        __global float *input,__global float *last_l)
    {
      int row = get_global_id(0);
      float sum = 0.0f;
      for(int i=0; i<colw; i++)
      {
           sum += (input[i] * wei[row*colw + i]);
      }
      float value = sum + bi[row];
      last_l[row] = value;
      
    }
    """).build()

costprog = cl.Program(ctx, """
    __kernel void cost(int roww, __global float *tar,__global float *last_l,__global float *cost)
    {
      int row = get_global_id(0);
      float diff =  last_l[row] - tar[row]  ;
      cost[row] = diff  ;
    
    }
    """).build()

subtract = cl.Program(ctx, """
    __kernel void sub(int colw, __global float *to_sub, __global float *input,__global float *cost )
    {
      int row = get_global_id(0);
      int col = get_global_id(1);
      to_sub[row*colw + col] = input[col] * cost[row];
    }
    """).build()

rate = cl.Program(ctx, """
    __kernel void learn( int colw,__global float *to_sub,__global float *new_sub)
    {
      int row = get_global_id(0);
      int col = get_global_id(1);
      new_sub[row*colw + col] = 0.01 * to_sub[row*colw + col] ;
    }
    """).build()

tune_weights = cl.Program(ctx, """
    __kernel void paraw(int colw, __global float *wei,__global float *to_sub,__global float *new_wei)
    {
      int row = get_global_id(0);
      int col = get_global_id(1);
      new_wei[row*colw + col] = wei[row*colw + col] - to_sub[row*colw + col];
    }
    """).build()

tune_bias = cl.Program(ctx, """
    __kernel void parab( __global float *bias,__global float *cost,__global float *new_bias)
    {
      int row = get_global_id(0);
      new_bias[row] = bias[row] - 0.01 * cost[row];
    }
    """).build()


def start(event):
    global weights,bias,inputimg ,inputim ,predimg ,predim ,expectimg ,expectim 
    z = 2
    fit = 1000
    for j in range(10):
        for i in range(1):
            arr1 = training_inputs[i]
            arr1= arr1.reshape(row,col)
            arr1 = arr1*fit
            inputim = Image.fromarray(arr1)
            w,h = inputim.size
            inputim = inputim.resize((w*z,h*z))
            inputimg = ImageTk.PhotoImage(inputim)
            c[0].create_image(0,0,anchor=tk.NW, image = inputimg)
            arr2 = target_outputs[i]
            arr2 = arr2.reshape(row,col)
            arr2 = arr2*fit
            expectim = Image.fromarray(arr2)
     #       print("ra",expectim.size)
            expectim = expectim.resize((w*z,h*z))
            expectimg = ImageTk.PhotoImage(expectim)
            c[2].create_image(0,0,anchor=tk.NW, image = expectimg)
            
            
            weights = weights.astype(np.float32)
            bias = bias.astype(np.float32)
            train_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=training_inputs[i])
            wei_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
            bias_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf= bias)
      #      print("last")
            last_l = np.zeros(row*col).astype(np.float32)
            last_l_buf = cl.Buffer(ctx, mf.WRITE_ONLY, last_l.nbytes)
            
            out.output(queue, last_l.shape, None,np.uint32(row*col),np.uint32(row*col),wei_buf,bias_buf,train_buf,last_l_buf)
            cl.enqueue_copy(queue,last_l, last_l_buf)
            predim = last_l.copy()
            print(last_l)
            print("Stage 1 ")
            predim = predim.reshape(row,col) 
            print(target_outputs[i][0])
            print(predim.shape,predim[0][0])
     #       print("ASD",target_outputs[i][row*col-1])
     #       print(last_l[row*col-1]) 
            predim = predim*fit
            predim = predim.astype(int)
            predim = Image.fromarray(predim)
            predim = predim.resize((w*z,h*z))
            predimg = ImageTk.PhotoImage(predim)
            c[1].create_image(0,0,anchor=tk.NW, image = predimg)
            root.update()
        #    print("LAST",last_l)
            train_buf.release()
            wei_buf.release()
            bias_buf.release()
            last_l_buf.release()
            error = np.zeros(row*col).astype(np.float32)
            tar_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_outputs[i])
            last_l_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=last_l)
            err_buf = cl.Buffer(ctx, mf.WRITE_ONLY, error.nbytes)
            costprog.cost(queue, last_l.shape, None,np.uint32(row*col),tar_buf,last_l_buf,err_buf)
            cl.enqueue_copy(queue,error, err_buf)
            print("Stage 2 ")
            tar_buf.release()
            last_l_buf.release()
            err_buf.release()
            
            error = error.astype(np.float32)            
            train_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=training_inputs[i])
            err_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=error)
            temp_sub = np.empty_like(weights).astype(np.float32) 
            temp_sub_buf = cl.Buffer(ctx, mf.WRITE_ONLY, temp_sub.nbytes)
            subtract.sub(queue, weights.shape, None,np.uint32(row*col),temp_sub_buf,train_buf,err_buf)
            cl.enqueue_copy(queue,temp_sub, temp_sub_buf)
            train_buf.release()
            err_buf.release()
            temp_sub_buf.release()            
        #    print("Stage 3 ",temp_sub)
            
            temp_sub_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=temp_sub)
            new_sub = np.empty_like(temp_sub).astype(np.float32) 
            new_sub_buf = cl.Buffer(ctx, mf.WRITE_ONLY, new_sub.nbytes)
            rate.learn(queue, weights.shape, None,np.uint32(row*col),temp_sub_buf,new_sub_buf)
            cl.enqueue_copy(queue,temp_sub, new_sub_buf)
            temp_sub_buf.release()            
            new_sub_buf.release()            
        #    print("stage 4",temp_sub)
            temp_sub_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=temp_sub)
            wei_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
            new_wei = np.empty_like(weights).astype(np.float32) 
            new_wei_buf = cl.Buffer(ctx, mf.WRITE_ONLY, new_wei.nbytes)
            tune_weights.paraw(queue, weights.shape, None,np.uint32(row*col),wei_buf,temp_sub_buf,new_wei_buf)
            cl.enqueue_copy(queue,weights, new_wei_buf)
            temp_sub_buf.release()
            wei_buf.release()
            new_wei_buf.release()
            
            bias_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf= bias)
            new_bias = np.empty_like(bias).astype(np.float32) 
            new_bias_buf = cl.Buffer(ctx, mf.WRITE_ONLY, new_bias.nbytes)
            err_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=error)
            tune_bias.parab(queue, bias.shape, None,bias_buf,err_buf,new_bias_buf)
            cl.enqueue_copy(queue,bias, new_bias_buf)
            err_buf.release()
            bias_buf.release()
            new_bias_buf.release()
            
        if (j+1)%10== 0 :
            start = time.time()
            np.save("weightsGPU.npy", weights, allow_pickle=True, fix_imports=True)
            start = time.time() - start
            print("Save weight ",start)            
            np.save("biasGPU.npy", bias, allow_pickle=True, fix_imports=True)
#            np.savetxt("weightsGPU.csv", weights, delimiter=",")
#            np.savetxt("biasGPU.csv", bias, delimiter=",")
         #   time.sleep(3)
            print("save")            
            print("done")
    
train = tk.Button(root, text='Train',width =5)
train.grid(column = 1, row = 1)
train.bind('<Button-1>', start)   

root.mainloop()
