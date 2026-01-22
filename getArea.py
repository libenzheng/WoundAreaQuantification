#!/usr/bin/env python
# coding: utf-8

# In[1]:


import imageio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy import stats
from scipy import ndimage
import os
import datetime
from matplotlib.colors import ListedColormap 


# In[ ]:


def get_area_from_image(filename,_path_color = [255,0,255],_area_threshold = 500,
                        switch_export_pdf = True,
                       switch_display = False):

    #import image
    _im = imageio.imread(filename)

    #color filtering to elucidate contour
    _image_size = [_im.shape[0],_im.shape[1]]


    _im_R = _im[:,:,0]
    _im_G = _im[:,:,1]
    _im_B = _im[:,:,2]

    _R = np.where(_im_R >= _path_color[0]-5,1, 0)
    _G = np.where(_im_G <= _path_color[1]+5,1, 0)
    _B = np.where(_im_B >= _path_color[2]-5,1, 0)

    _color_filtered_im = _R*_G*_B


    #build mask to fill contours
    _masked_im = ndimage.morphology.binary_fill_holes(_color_filtered_im)

    #     plt.imshow(_masked_im)

    #label all masks
    _label_im, _nb_label = ndimage.label(_masked_im)

    #get data list

    _area_list = []
    _x_list = []
    _y_list = []
    _xy_center_list = []

    for _i_label in range(1,_nb_label+1):
        _mask = _label_im == _i_label
        _area = _mask.sum()#in number of pixels

        if _area>=_area_threshold:
            _x,_y = ndimage.measurements.center_of_mass(_mask)
            _area_list.append(_area)
            _x_list.append(_x)
            _y_list.append(_y)

            _x_sum = _mask.sum(axis = 0)
            _x_center = np.arange(len(_x_sum))[_x_sum==_x_sum.max()].mean()
            _y_sum = _mask.sum(axis = 1)
            _y_center = np.arange(len(_y_sum))[_y_sum==_y_sum.max()].mean()
            _xy_center_list.append([_x_center,_y_center])

    _area_list = np.array(_area_list)
    _x_list = np.array(_x_list)
    _y_list = np.array(_y_list)



    if len(_area_list)>5:
        print('Image more than 5 sites with areas >= ',_area_threshold, 'pxls, img: ',filename)



    # location identification
    _number_of_locatins = 5 
    _reference_label = 5
    _location = np.ones(_number_of_locatins)*-1

    #find the middle one 
    _x_diff = _x_list - _x_list.mean()
    _y_diff = _y_list - _y_list.mean()
    _distance = abs(_x_diff) + abs(_y_diff)
    _ref_at_list = np.arange(len(_distance))[_distance == _distance.min()]
    _location[_ref_at_list] = _reference_label

    #find corresponding relative locations 
    _x_diff_ref = np.sign(_x_list - _x_list[_ref_at_list])
    _y_diff_ref = np.sign(_y_list - _y_list[_ref_at_list])

    _i_seq = 0 
    for _x,_y in zip(_x_diff_ref,_y_diff_ref):
        if (_x==-1)&(_y==-1):
            _location[_i_seq]=1
        if (_x==1)&(_y==-1):
            _location[_i_seq]=2
        if (_x==-1)&(_y==1):
            _location[_i_seq]=3
        if (_x==1)&(_y==1):
            _location[_i_seq]=4
        _i_seq+=1


    # get real area
    _ref_area_pixel = _area_list[_ref_at_list]
    _ref_area_real = 3*3*np.pi # in mm^2
    _area_real_list = np.zeros(len(_area_list))
    for _i,_area in enumerate(_area_list):
        _area_real = _area/_ref_area_pixel*_ref_area_real
        _area_real_list[_i] = _area_real


    #export data
    _export_data = np.zeros(_number_of_locatins-1)
    for _i_export in range(1,_number_of_locatins):
        _data_at_list = np.arange(len(_location))[_location == _i_export]
        _area_real = _area_real_list[_data_at_list]
        if _i_export!=_reference_label:
            _export_data[_i_export-1]+=_area_real

    # plot

    if switch_export_pdf:

        #plot bg image
        fig,axs = plt.subplots(1,3,dpi=300)
        _suptitle = '.'.join(filename.split('.')[:-1])
        _suptitle = _suptitle.replace('\\',' | ')
        # fig.suptitle(_suptitle,fontsize=_font_size)

        # rwa image
        ax = axs[0]
        ax.set_axis_off()
        _raw_name = filename.split('.')
        _raw_name[-1] = 'jpg'
        _raw_name = '.'.join(_raw_name)
        _im_raw = imageio.imread(_raw_name)
        ax.imshow(_im_raw)
        # ax.set_title('Raw image',fontsize=_font_size)


        # rwa image
        ax = axs[1]
        ax.set_axis_off()
        ax.imshow(_im)
        ax.set_title(_suptitle)


        # processed image
        ax.imshow(_im)
        ax = axs[2]
        ax.set_axis_off()
        ax.imshow(_im)
        # ax.set_title('Processed image',fontsize=_font_size)

        _im_mask = _label_im.copy()
        _im_mask[_im_mask>=1]=1



        # Choose colormap
        cmap = plt.cm.gray
        # Get the colormap colors
        my_cmap = cmap(np.arange(cmap.N))
        # Set alpha
        my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
        my_cmap = ListedColormap(my_cmap)


        plt.imshow(_im_mask,cmap = my_cmap,vmax = 1,vmin = 0,alpha = 0.75)

        # annotation
        for _xy_center,_area_real,_area,_location_label in zip(_xy_center_list,_area_real_list,_area_list,_location):

            _str_real_area = str(np.round(_area_real,2)) +'$ mm^2$ '
            _str_area = str(int(_area/1000))+'k pixels'

            _str_location = 'Wound #'+str(int(_location_label)) if _location_label!=5 else 'Reference'
            _str =  _str_location+'\n'+_str_real_area+'\n' +_str_area

            ax.text(_xy_center[0],_xy_center[1],_str,
                    fontsize = 3,
                    bbox = dict(facecolor='white', 
                                  edgecolor='black',
                                  boxstyle='round,pad=1',
                                 alpha=0.4))

        #export figure

        _export_name = filename.split('.')
        # _export_name[-2]+='_area'
        _export_name[-1] = 'pdf'
        _export_name = '.'.join(_export_name)
        plt.tight_layout()
        plt.savefig(_export_name)
        
        if not switch_display:
            plt.close(fig);
        
        
    return _export_data



# In[ ]:


#basic function
def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir) 
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


def sort_file_list(path_list,sort_name):
    _selected_path_list = []
    for path_i in path_list:
        pathname = path_i
        if pathname.endswith(sort_name):
            _selected_path_list.append(pathname)
        
    return _selected_path_list

