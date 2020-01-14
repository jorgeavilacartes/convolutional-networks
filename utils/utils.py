import numpy as np 


def Conv2D(input_tensor, kernel, stride, print_dims = True):
    """Naive approach to convolution between and 2D input_tensor and a 2D filter/kernel
    
    Arguments:
        input_tensor {numpy.array} -- Input tensor to the convolution, by default it only accept depth equals to 1. 
                                      It will be trated as a (height, width) image. If the input have (height, width,channel) 
                                      dimensions, it will be rescaled to two dimension (height, width)

        kernel {numpy.array} -- filter to be applied to the input_tensor
        stride {int or tuple} -- horizontal and vertical displacement
    
    Keyword Arguments:
        print_dims {bool} -- If True, then dimensions of input, output, kernel and stride will be printed (default: {True})
    
    Returns:
        [numpy.array] -- The resulting tensor after convolve the input_tensor with the kernel
    """    
    # Dimension input
    assert(len(input_tensor.shape) in set([1,2])), "input_tensor must have dimension 2 or 3. Yours have dimension {}".format(len(input_tensor))

    if input_tensor.shape == 3:
        input_tensor = input_tensor[:,:,0]

    # Stride: desplazamiento horizontal y vertical
    if isinstance(stride,int):
        sh, sw = stride, stride
    elif isinstance(stride,tuple):
        sh, sw = stride
    
    # Dimensiones del input (height, width)
    n_ah, n_aw = input_tensor.shape
    
    # Dimension del filtro o kernel (nk,nk) SIEMPRE es cuadrado
    n_k  = kernel.shape[0] 
    
    dim_out_h = int(np.floor( (n_ah - n_k) / sh + 1 ))
    dim_out_w = int(np.floor( (n_aw - n_k) / sw + 1 ))
    
    # Inicializar el output
    output = np.zeros([dim_out_h, dim_out_w])
    
    start_row = 0
    for i in range(dim_out_h):
        # Cuando parta una nueva fila, volver a la primera columna
        start_col = 0
        for j in range(dim_out_w):
            
            # Aplicamos la operacion entre tensores
            sub_tensor = input_tensor[start_row:(start_row+n_k), start_col:(start_col+n_k)]
            
            #Actualizar componente del output
            output[i, j] = np.tensordot(sub_tensor , kernel)
            
            #Avanzar stride horizontal
            start_col += sw
            
        # Avanzar stride vertical
        start_row += sh
        
    if print_dims: 
        print("- Input tensor dimensions", input_tensor.shape)
        print("- Kernel dimensions", kernel.shape)
        print("- Stride (h,w) ", (sh, sw))
        print("- Convolved tensor dimension", output.shape)
        
    return output


def MaxPooling2D(input_tensor, size_kernel, stride, print_dims = True):
    """[summary]
    
    Arguments:
        input_tensor {numpy.array} -- Input tensor to the convolution, by default it only accept depth equals to 1. 
                                      It will be trated as a (height, width) image. If the input have (height, width,channel) 
                                      dimensions, it will be rescaled to two dimension (height, width)
        size_kernel {int} -- size of kernel to be applied. Usually 3,5,7. It means that a kernel of (size_kernel, size_kernel) will be applied
                             to the image.
        stride {int or tuple} -- horizontal and vertical displacement
    
    Keyword Arguments:
        print_dims {bool} -- [description] (default: {True})
    
    Returns:
        [numpy.array] -- The resulting tensor after apply the MaxPooling operation to the input_tensor
    """   
    
    # Dimension input
    assert(len(input_tensor.shape) in set([1,2])), "input_tensor must have dimension 2 or 3. Yours have dimension {}".format(len(input_tensor))

    if input_tensor.shape == 3:
        input_tensor = input_tensor[:,:,0]

    # Stride: desplazamiento horizontal y vertical
    if isinstance(stride,int):
        sh, sw = stride, stride
    elif isinstance(stride,tuple):
        sh, sw = stride
    
    # Dimensiones del input (height, width)
    n_ah, n_aw = input_tensor.shape
    
    # Dimension del filtro o kernel (nk,nk) SIEMPRE es cuadrado
    n_k  = size_kernel
    
    dim_out_h = int(np.floor( (n_ah - n_k) / sh + 1 ))
    dim_out_w = int(np.floor( (n_aw - n_k) / sw + 1 ))
    
    # Inicializar el output
    output = np.zeros([dim_out_h, dim_out_w])
    
    start_row = 0
    for i in range(dim_out_h):
        start_col = 0
        for j in range(dim_out_w):
            
            # Aplicamos la operacion entre tensores
            sub_tensor = input_tensor[start_row:(start_row+n_k), start_col:(start_col+n_k)]
            #print(sub_tensor)
            output[i, j] = np.max(sub_tensor)
            
            start_col += sw
        start_row += sh
        
    if print_dims: 
        print("- Input tensor dimensions", input_tensor.shape)
        print("- Kernel dimensions", (n_k, n_k))
        print("- Stride (h,w) ", (sh, sw))
        print("- Convolved tensor dimension", output.shape)
        
    return output


import matplotlib.pyplot as plt
from scipy.signal import convolve2d, convolve

def show_two_plots(img1, img2, title_img1 = 'Original image', title_img2 = 'Convoluted image'):
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.title(title_img1)
    plt.axis('off')
    plt.imshow(img1, cmap='gray')

    plt.subplot(122)
    plt.title(title_img2)
    plt.axis('off')
    plt.imshow(img2, cmap='gray')
    
def show_differences(img, kernel, conv_func = "scipy", stride = None ):
    if conv_func == "own":
        if stride == None:
            stride = 1
        convolved = Conv2D(img, kernel, stride)
    else:
        convolved = convolve2d(img, kernel, mode = "valid")
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.title('Input image')
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
    plt.subplot(122)
    plt.title('Convolved image')
    plt.axis('off')
    plt.imshow(convolved, cmap='gray')
    
    print("Original image dimensions", img.shape)
    print("Kernel dimensions", kernel.shape)
    print("Convolved image dimensions", convolved.shape)
    
    return convolved