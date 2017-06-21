import argparse
import datetime
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.utils as nnutils
from torch.autograd import Variable
from libs.import_utils import *
from nets.squeezenet import SqueezeNet
import numpy as np


# Define Arguments and Default Values
parser = argparse.ArgumentParser(description='PyTorch z2_color Training',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--validate', type=str, metavar='PATH',
                    help='path to model for validation')
# parser.add_argument('--skipfirstval', type=str, metavar='PATH',
#                     help='Skip the first validation (if restoring an end of epoch save file)')
parser.add_argument('--resume', type=str, metavar='PATH',
                    help='path to model for training resume')
parser.add_argument('--require-one', default=[], type=str, nargs='+',
                    help='Mandatory run labels, runs without these labels will be ignored.')
parser.add_argument('--cuda-device', default=0, type=int, help='Cuda GPU ID to use for GPU Acceleration.')
parser.add_argument('--batch-size', default=64, type=int, help='Number of datapoints in a mini-batch for training.')
parser.add_argument('--saverate', default=700, type=int,
                    help='Number of batches after which a progress save is done.')
args = parser.parse_args()

img_width = 672.
img_height = 376.

if args.resume is not None:
    save_data = torch.load(args.resume)
    start_ctrl_low = save_data['low_ctr']
    start_ctrl_high = save_data['high_ctr']


def instantiate_net():
    net = SqueezeNet().cuda()
    criterion = nn.MSELoss().cuda()  # define loss function
    optimizer = torch.optim.Adadelta(net.parameters())
    return net, criterion, optimizer

def load_car_image():
    # TODO: Create first a list/datastructure to draw from
    # TODO: Remove absolute fixed paths
    
    base_dir = '/media/picard/f765bbb1-29a1-4839-b2e3-2afa2bbb9d34/carData/car'
    filename = random.choice(os.listdir(base_dir))
    
    return cv2.imread(os.path.join(base_dir,filename))
    
def load_no_car_image():
    # TODO: Create first a list/datastructure to draw from
    base_dir = '/media/picard/f765bbb1-29a1-4839-b2e3-2afa2bbb9d34/carData/nocar'
    filename = random.choice(os.listdir(base_dir))
    
    return cv2.imread(os.path.join(base_dir,filename))

def pick_data():
    
    if(np.random > 0.5):
        return_image = load_car_image()
    else:
        return_image = load_no_car_image()

    # Change code to select car / no car instead of low and high steer
    assert(return_image != None)
    return return_image


def pick_validate_data():

    # TODO: Create actual different dataset for validation    
    if(np.random > 0.5):
        return_image = load_car_image()
    else:
        return_image = load_no_car_image()

    # Change code to select car / no car instead of low and high steer
    assert(return_image != None)
    return return_image


def get_camera_data(data):
    listoftensors = []
    
    listoftensors.append(torch.from_numpy(data))

    camera_data = torch.cat(listoftensors, 2)
    camera_data = camera_data.cuda().float()/255. - 0.5
    # Switch dimensions to match neural net
    camera_data = torch.transpose(camera_data, 0, 2)
    camera_data = torch.transpose(camera_data, 1, 2)

    return camera_data



def get_labels(data):
    # TODO: Add the second net here and do this now only for testing purposes
    
    x1 = np.random.random() * img_width
    y1 = np.random.random() * img_height
    x2 = np.random.random() * img_width
    y2 = np.random.random() * img_height
    
    bounding_box = torch.cuda.FloatTensor([[x1,y1],[x2,y2]])

    return bounding_box


def get_batch_data(batch_size, data_function):
    
    batch_input = torch.FloatTensor().cuda()
    batch_labels = torch.FloatTensor().cuda()
    # TODO: implement real progress indicator
    progress = 0

    for batch in range(batch_size):  # Construct batch
        data = None
        while 'data' not in locals() or data is None:
            data = data_function()

        if data == None:  # If out of data, return done and skip batch
            return progress, False, None, None, None

        camera_data = get_camera_data(data)
        labels = get_labels(data)

        # Creates batch
        batch_input = torch.cat((torch.unsqueeze(camera_data, 0), batch_input), 0)
        batch_labels = labels#torch.cat((torch.unsqueeze(labels, 0), batch_labels), 0)

    # Train and Backpropagate on Batch Data
    return progress, True, batch_input, batch_labels


torch.set_default_tensor_type('torch.FloatTensor')  # Default tensor to float for consistency

torch.cuda.set_device(args.cuda_device)  # Cuda device ID

# Load Data
# TODO change low steer high steer
#random.shuffle(low_steer)
#random.shuffle(high_steer)
#low_steer_train = low_steer[:int(0.9*len(low_steer))]
#high_steer_train = high_steer[:int(0.9*len(high_steer))]
#low_steer_val = low_steer[int(0.9*len(low_steer)):]
#high_steer_val = high_steer[int(0.9*len(high_steer)):]

net, criterion, optimizer = instantiate_net()

cur_epoch = 0
# if args.resume is not None:
#     save_data = torch.load(args.resume)
#     net.load_state_dict(save_data['net'])
#     optimizer.load_state_dict(save_data['optim'])
#     cur_epoch = save_data['epoch']
#     
#     # load shuffled datasets
#     low_steer_train = save_data['lst']
#     high_steer_train = save_data['hst']
#     low_steer_val = save_data['lsv']
#     high_steer_val = save_data['hsv']
# if args.validate is not None:
#     save_data = torch.load(args.validate)
#     net.load_state_dict(save_data['net'])
# 
#     sum = 0
#     count = 0
#     notFinished = True  # Checks if finished with dataset
#     net.eval()
#     while notFinished:
#         random.shuffle(low_steer_val)
#         random.shuffle(high_steer_val)
#         # Load batch
#         progress, notFinished, batch_input, batch_metadata, batch_labels = get_batch_data(1, pick_validate_data)
#         if not notFinished:
#             break
# 
#         # Run neural net + Calculate Loss
#         outputs = net(Variable(batch_input), Variable(batch_metadata)).cuda()
#         print(outputs)
#         print(batch_labels)
# 
#         loss = criterion(outputs, Variable(batch_labels))
#         count += 1
#         sum += loss.data[0]
# 
#         # print('Output:\n' + str(outputs) + '\nLabels:\n' + str(batch_labels))
#         print('Average Loss: ' + str(sum / count))
# else:
if True: #TODO remove this line and add validation again
    print(net)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file = open('logs/log_file' + str(datetime.datetime.now().isoformat()), 'w')
    log_file.truncate()

    for epoch in range(cur_epoch, 1000000):  # Iterate through epochs
        cur_epoch = epoch
        # Training
        notFinished = True  # Checks if finished with dataset
        #random.shuffle(low_steer_train)
        #random.shuffle(high_steer_train)
        
        batch_counter = 0
        sum = 0
        sum_counter = 0
        start = time.time()
        net.train()
        while notFinished:
            # Load batch
            progress, notFinished, batch_input, batch_labels = get_batch_data(args.batch_size, pick_data)
            if not notFinished:
                break

            # zero the parameter gradients
            optimizer.zero_grad()

            # Run neural net + Calculate Loss
            
            outputs = net(Variable(batch_input)).cuda()
            loss = criterion(outputs, Variable(batch_labels))

            # Backprop
            loss.backward()
            nnutils.clip_grad_norm(net.parameters(), 1.0)
            optimizer.step()

            # Update progress bar
            # deleted animate
            batch_counter += 1
            sum_counter += 1
            sum += loss.data[0]
            
            if sum_counter == 10:
                print (str(batch_counter) + ',' + str(sum / sum_counter) + "," + str(outputs.data))
                
                sum = 0
                sum_counter = 0
        sum = 0
        count = 0
        notFinished = True  # Checks if finished with dataset
        
        net.eval()
        while notFinished:
            # Load batch
            progress, notFinished, batch_input, batch_metadata, batch_labels = get_batch_data(1, pick_validate_data)

            if not notFinished:
                break

            # Run neural net + Calculate Loss
            outputs = net(Variable(batch_input), Variable(batch_metadata)).cuda()
            loss = criterion(outputs, Variable(batch_labels))
            
            count += 1
            sum += loss.data[0]

            

            if count % 1000 == 0:
                pb.animate(progress)
                log_file.write('\nAverage Validation Loss,' + str(sum / count))
                log_file.flush()

        log_file.write('\nFinish cross validation! Average Validation Error = ' + str(sum / count))
        log_file.flush()
        save_data = {'low_ctr': 0, 'high_ctr': 0, 'net': net.state_dict(),
                     'optim': optimizer.state_dict(), 'epoch': cur_epoch,
                     'lst':low_steer_train, 'hst':high_steer_train,
                     'lsv':low_steer_val,'hsv':high_steer_val
                    }
        if not os.path.exists('save'):
            os.makedirs('save')
        torch.save(save_data, 'save/epoch_save_' + str(cur_epoch) + '.' + str(sum / count))
    