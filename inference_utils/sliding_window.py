import torch, warnings, os
import numpy as np

def gpu_swi(image, model, n_classes, window_size=(1, 112, 176, 176), batch=False):
    """
    Sliding Window
    Args:
        image (tensor): shape (B, C, D, H, W)
        model (Pytorch Model)
        n_classes (int): Number of classes to segment
        window_size (tuple): shape of window size
        mira_flag (bool): controls how many windows to use
        batch (bool): some models don't take batch dimension
    Returns:
        tensor: The segmented output
    Example:
    >>> PATH = '/content/drive/MyDrive/BHKLAB/Mira_Inference'
    >>> model = WolnyUNet3D(num_classes=20, f_maps=48)
    >>> test = torch.zeros(1, 128, 224, 224)
    >>> res = optimized_swi(test, model, 20, PATH, mira_flag=True)
    """
    if len(image.shape) < 5:
        image = image.unsqueeze(0)

    br, zr, yr, xr = window_size
    _, _, z, y, x = image.shape

    window_z_coords = [(0, zr), (z//2-zr//2, z//2+zr//2), (z-zr, z)]
    window_y_coords = [(0, yr), (y-yr, y)]
    window_x_coords = [(0, xr), (x-xr, x)]


    output_shape = (br, n_classes, z, y, x)
    window_accumalator = torch.zeros(output_shape).to(image.device)
    window_reference = torch.zeros(output_shape).to(image.device)
    
    with torch.inference_mode():
      for start_z, end_z in window_z_coords:
          for start_y, end_y in window_y_coords:
              for start_x, end_x in window_x_coords:
                  if batch:
                      window = image[:, :, start_z:end_z, start_y:end_y, start_x:end_x].to(image.device)
                  else:
                      window = image[:, :, start_z:end_z, start_y:end_y, start_x:end_x][0].to(image.device) # Removing batch dimension
                  if (not batch and window.shape != window_size) or (batch and window[0].shape != window_size):
                      pass
                  else:
                      window_accumalator[:, :, start_z:end_z, start_y:end_y, start_x:end_x] +=  model(window) 
                      window_reference[:, :, start_z:end_z, start_y:end_y, start_x:end_x] += 1
    
    return window_accumalator/window_reference

def optimized_swi(image, model, n_classes, window_dir, window_size=(1, 112, 176, 176), batch=False):
    """
    Sliding Window optimized for CPU. Ensembles prediction from models if multiple are inputed.
    Args:
        image (tensor): shape (B, C, D, H, W)
        model (Pytorch Model)
        n_classes (int): Number of classes to segment
        window_dir (str): directory to temporarily store windows
        window_size (tuple): shape of window size
        mira_flag (bool): controls how many windows to use
    Returns:
        tensor: The segmented output
    Example:
    >>> PATH = '/content/drive/MyDrive/BHKLAB/Mira_Inference'
    >>> model = WolnyUNet3D(num_classes=20, f_maps=48)
    >>> test = torch.zeros(1, 128, 224, 224)
    >>> res = optimized_swi(test, model, 20, PATH, mira_flag=True)
    """
    
    if len(image.shape) < 5:
        image = image.unsqueeze(0)

    br, zr, yr, xr = window_size
    _, _, z, y, x = image.shape

    window_z_coords = [(0, zr), (z//2-zr//2, z//2+zr//2), (z-zr, z)]
    window_y_coords = [(0, yr), (y-yr, y)]
    window_x_coords = [(0, xr), (x-xr, x)]


    output_shape = (br, n_classes, z, y, x)
    num_windows = 0
    window_coords = []

    model.eval()
    torch.no_grad()

    for start_z, end_z in window_z_coords:
        for start_y, end_y in window_y_coords:
            for start_x, end_x in window_x_coords:
                if batch:
                    window = image[:, :, start_z:end_z, start_y:end_y, start_x:end_x].to(image.device)
                else:
                    window = image[:, :, start_z:end_z, start_y:end_y, start_x:end_x][0].to(image.device) # Removing batch dimension
                if (not batch and window.shape != window_size) or (batch and window[0].shape != window_size):
                    pass
                else:
                    out = model(window)
                    torch.save(out, os.path.join(window_dir, f'window_{num_windows}'))
                    del out
                    num_windows += 1
                    window_coords.append([(start_z, end_z), (start_y, end_y), (start_x, end_x)])

    print(f"{num_windows} created, accumalating...")
    window_accumalator = torch.zeros(output_shape).to(image.device)
    window_reference = torch.zeros(output_shape).to(image.device)

    for idx in range(num_windows):
        start, end = zip(*window_coords[idx])
        start_z, start_y, start_x = start
        end_z, end_y, end_x = end

        window = torch.load(os.path.join(window_dir, f'window_{idx}'))
        window_accumalator[:, :, start_z:end_z, start_y:end_y, start_x:end_x] += window
        window_reference[:, :, start_z:end_z, start_y:end_y, start_x:end_x] += 1
        del window

    return window_accumalator/window_reference

def swi(image, net, n_classes, roi_size=(2, 112, 176, 176), mira_flag:bool()=False):

    # in our case we're looking for a 5D tensor...
    if len(image.size()) < 5:
        image = image.unsqueeze(0)
    
    br,zr,yr,xr = roi_size
    # change shape to individual variables
    b,c,z,y,x = image.size()
    warnings.warn(f'Image shape is {image.size()}')

    if mira_flag is True:
        # Z ... for MIRA
        warnings.warn("Using sliding MIRA optimized for cpu(s)")
        start_z = [0, z//2-zr//2, z-zr]
        end_z = [zr, z//2+zr//2, z]
        # Y
        start_y = [0, y-yr]
        end_y = [yr, y]
        # X
        start_x = [0, x-xr]
        end_x = [xr, x]
    else:
        # Used for testing during study...
        # Z ...
        start_z = [0, z//4, z-z //4 - zr, z-zr]
        end_z = [zr, z//4+zr, z-z//4, z]
        # Y
        start_y = [0, y-yr, y//12, y//6, y - y//4 - yr, y - y//6 - yr]
        end_y = [yr, y, y//12 + yr, y//6 + yr, y - y//4, y - y//6]
        # X
        start_x = [0, x-xr, x//12, x//4, x//6, x - x//4 - xr, x - x//6 - xr, x - x//12 - xr]
        end_x = [xr, x, x//12 + xr, x//4+xr, x //6 + xr, x - x//4, x - x//6, x - x//12]
    
    output_shape = (br, n_classes, z, y, x)

    reference_ = torch.zeros(output_shape).to(image.device)
    reference = torch.zeros(output_shape).to(image.device)
    iter_=0
    net.eval()
    torch.no_grad()
    for i, val in enumerate(start_z):
        for j, v in enumerate(start_y):
            for k, va in enumerate(start_x):
                im = image[:,:,val:end_z[i], v:end_y[j], va:end_x[k]]
                sh = im.size() # shoud be 5D tensor...
                if (sh[1], sh[2], sh[3], sh[4]) != roi_size:
                    warnings.warn(f'Image shape is {im.size()}...passing step...')
                    pass
                else:
                    reference_[:,:,val:end_z[i], v:end_y[j], va:end_x[k]]+=1
                    # RUN THE NETWORK
                    im=im[0]
                    warnings.warn(f'NET IN SIZE IS {im.size()}')
                    start = time.time()
                    output=net(im)
                    end = time.time()
                    print(end-start, " seconds")
                    warnings.warn(f'NET OUTS SIZE IS {output.size()}')
                    output
                    reference[:,:, val:end_z[i], v:end_y[j], va:end_x[k]] += output
                    iter_ += 1
                    if iter_%20 == 0:
                        warnings.warn(f'Iterated {iter_} times with sliding window.')
                
                gc.collect()
    
    return reference/reference_

def ensemble_optimized_swi(image, models, n_classes, window_dir, window_size=(1, 112, 176, 176), mira_flag=False):
    """
    Sliding Window optimized for CPU. Ensembles prediction from models if multiple are inputed.
    Args:
        image (tensor): shape (B, C, D, H, W)
        models (Pytorch Models)
        n_classes (int): Number of classes to segment
        window_dir (str): directory to temporarily store windows
        window_size (tuple): shape of window size
        mira_flag (bool): controls how many windows to use
    Returns:
        tensor: The segmented output
    Example:
    >>> PATH = '/content/drive/MyDrive/BHKLAB/Mira_Inference'
    >>> model = WolnyUNet3D(num_classes=20, f_maps=48)
    >>> test = torch.zeros(1, 128, 224, 224)
    >>> res = optimized_swi(test, model, 20, PATH, mira_flag=True)
    """
    
    if not isinstance(models, list): # if only one model is passed in
        models = [models]

    if len(image.shape) < 5:
        image = image.unsqueeze(0)

    br, zr, yr, xr = window_size
    _, _, z, y, x = image.shape

    if mira_flag:
        window_z_coords = [(0, zr), (z//2-zr//2, z//2+zr//2), (z-zr, z)]
        window_y_coords = [(0, yr), (y-yr, y)]
        window_x_coords = [(0, xr), (x-xr, x)]
    else:
        window_z_coords = [(0, zr), (z//4-zr//4, z//4+zr//4), (z-zr, z)]
        window_y_coords = [(0, yr), (y-yr, y), (y//12, y//12 + yr), (y//6, y//6 + yr), ( y - y//4 - yr, y - y//4), ( y - y//6 - yr, y - y//6)]
        window_x_coords = [(0, xr), (x-xr, x), (x//12, x//12 + xr), (x//4, x//4 + xr), (x//6, x//6 + xr), ( x - x//4 - xr, x - x//4), ( x - x//12 - xr, x - x//6)]


    output_shape = (br, n_classes, z, y, x)
    num_windows = 0
    window_coords = []

    for model in models:
        model.eval()
    torch.no_grad()

    for start_z, end_z in window_z_coords:
        for start_y, end_y in window_y_coords:
            for start_x, end_x in window_x_coords:
                window = image[:, :, start_z:end_z, start_y:end_y, start_x:end_x][0] # Removing batch dimension
                if window.shape != window_size:
                    pass
                else:
                    ensembled_window = torch.zeros((br, n_classes, zr, yr, xr)).to(image.device)
                    for idx, model in enumerate(models):
                        out = model(window)
                        warnings.warn(f'model {idx}')
                        ensembled_window += out
                    del out
                    ensembled_window /= len(models)
                    torch.save(ensembled_window, os.path.join(window_dir, f'window_{num_windows}'))
                    del ensembled_window
                    num_windows += 1
                    window_coords.append([(start_z, end_z), (start_y, end_y), (start_x, end_x)])

    print(f"{num_windows} created, accumalating...")
    window_accumalator = torch.zeros(output_shape).to(image.device)
    window_reference = torch.zeros(output_shape).to(image.device)

    for idx in range(num_windows):
        start, end = zip(*window_coords[idx])
        start_z, start_y, start_x = start
        end_z, end_y, end_x = end

        window = torch.load(os.path.join(window_dir, f'window_{idx}'))
        window_accumalator[:, :, start_z:end_z, start_y:end_y, start_x:end_x] += window
        window_reference[:, :, start_z:end_z, start_y:end_y, start_x:end_x] += 1
        del window

    return window_accumalator/num_windows
