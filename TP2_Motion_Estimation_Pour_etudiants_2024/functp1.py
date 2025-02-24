import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.signal import convolve2d



def initVector(ref,cur):
    """
    Performs an initialization for the first regularizers

    Parameters
    ----------
    ref : np.array
        Reference image.
    cur : np.array
        Reference image.

    Returns
    -------
    pV : np.array (vector of size 2)
        Regularizer for displacement.

    """
    
    
    step = 8
    cont = 4*step
    
    REF = gaussian_filter(ref,1.) # Unclear how to set sigma
    CUR = gaussian_filter(cur,1.)
    
    CUR = CUR[cont+1:(np.shape(CUR)[0]-cont):step,cont+1:(np.shape(CUR)[1]-cont):step]
    SSDMIN = np.inf
    
    pV=np.zeros(2)
    
    for globR in range(-cont,cont):
        for globC in range(-cont,cont):
            RR = REF[cont+1-globR:(cont-globR+np.shape(CUR)[0]*step):step, cont+1-globC:(cont-globC+np.shape(CUR)[1]*step):step]
            SSD = np.sum((RR-CUR)**2)
            
            if SSD<SSDMIN:
                SSDMIN=SSD
                pV[0]=globR
                pV[1]=globC
                
                
    return pV


def computePredictor(r,c,brow,bcol,mvf,ref,cur):
    """
    compute predictor gives the median of the mvf of the blocks :
        - to the left of the current block
        - above the current block
        - upper left of the current block
        
    If such blocks do not exist due to the border effects, they are not taken into account.

    Parameters
    ----------
    See usage in the me_ssd function

    Returns
    -------
    pV : Median of the mvf of the neighboor blocks

    """
    if r < brow and c < bcol:
        pV = initVector(ref,cur)
        
    elif r < brow: # First row
        pV = mvf[r,c-bcol,:]
        
    elif c < bcol: # First column
        pV = mvf[r-brow,c,:]
        
    else: # Inside
        if c >= np.shape(mvf)[1]-bcol: # Last column
            vC = mvf[r-brow,c-bcol,:]
        
        else: # Not the last column
            vC = mvf[r-brow,c+bcol,:]
            
        vA = mvf[r,c-bcol,:]
        vB = mvf[r-brow,c,:]

        temp = np.array([vA, vB, vC]).T

        pV = np.median(temp,axis = 1)
        
    pV = pV.ravel()
    
    return pV



def me_ssd(cur, ref, brow, bcol, search, lamb=0):
    """
    ME BMA full search Motion estimation
    mvf, prediction = me_ssd(cur, ref, brow, bcol, search);

    A regularization constraint can be used
    mvf = me(cur, ref, brow, bcol, search, lambda);
    In this case the function minimize SAD(v)+lambda*error(v)
    where error(v) is the difference between the candidate vector v and the
    median of its avalaible neighbors.
 
    Code inspired from the one of Marco Cagnazzo


    Parameters
    ----------
    cur : numpy array
        Current (i.e. second) frame of the video.
    ref : numpy array
        Previous (i.e. first) frame of the video.
    brow : int
        Number of rows in each block.
    bcol : int
        Number of rows in each block.
    search : int
        Search radius
    lamb : double
        Regularization parameter

    Returns
    -------
    mvf : TYPE
        DESCRIPTION.
    prediction : TYPE
        DESCRIPTION.

    """
    
    extension = search
    
    ref_extended = cv2.copyMakeBorder(ref, extension, extension, extension, extension, cv2.BORDER_REPLICATE) # To avoid border effect
    prediction = np.zeros(np.shape(cur));
    lamb *= brow*bcol;

    mvf = np.zeros((np.shape(cur)[0],np.shape(cur)[1],2))
    
    # Non-regularized search
    if lamb == 0.:
        for r in range(0,cur.shape[0],brow): # for each block in the current image, find the best corresponding block in the reference image
            for c in range(0,cur.shape[1],bcol):
                # current block selection
                B = cur[r:r+brow,c:c+bcol] # Block

                # Initialization:
                bestV = [0,0]
                costMin = np.inf
                
                Rbest = np.zeros((brow,bcol))
                
                # Loop on candidate displacement vectors
                for dcol in range(-extension,extension): # dcol = candidate displacement vector over the columns
                    for drow in range(-extension,extension):# drow = candidate displacement vector over the rows
                        A = ref_extended[r+drow+extension:r+drow+extension+brow,c+dcol+extension:c+dcol+extension+bcol]
                        cost = np.sum((B-A)**2)
                        
                        if cost < costMin: # Save the results if they are better than the previous ones
                            costMin = cost
                            Rbest = A
                            bestV = [drow,dcol]
    
                mvf[r:r+brow,c:c+bcol,0] = bestV[0]*np.ones((brow,bcol)) # Once the loop is over, save the best row displacement field
                mvf[r:r+brow,c:c+bcol,1] = bestV[1]*np.ones((brow,bcol))# Once the loop is over, save the best column displacement field
                prediction[r:r+brow,c:c+bcol]= Rbest
                   
    else: # Regularized search
        for r in range(0,cur.shape[0],brow): # for each block in the current image, find the best corresponding block in the reference image
            for c in range(0,cur.shape[1],bcol):
                # current block selection
                B = cur[r:r+brow,c:c+bcol] # Block

                # Initialization:
                bestV = [0,0]
                costMin = np.inf
                
                Rbest = np.zeros((brow,bcol))

                # Neighbours : pV is the regularization vector. The regularizer must be such that the estimated displacement is not too far away from pV
                pV = computePredictor(r,c,brow,bcol,mvf,ref,cur)
                
                # Loop on candidate displacement vectors
                for dcol in range(-extension,extension): # dcol = candidate displacement vector over the columns
                    for drow in range(-extension,extension):# drow = candidate displacement vector over the rows
                        A = ref_extended[r+drow+extension:r+drow+extension+brow,c+dcol+extension:c+dcol+extension+bcol]
                        cost = np.sum((B-A)**2)+lamb*np.linalg.norm(pV-np.array([drow,dcol])) # here, encore the REGULARIZED cost function
                        
                        if cost < costMin: # Save the results if they are better than the previous ones
                            costMin = cost
                            Rbest = A
                            bestV = [drow,dcol]
    
                mvf[r:r+brow,c:c+bcol,0] = bestV[0]*np.ones((brow,bcol)) # Once the loop is over, save the best row displacement field
                mvf[r:r+brow,c:c+bcol,1] = bestV[1]*np.ones((brow,bcol))# Once the loop is over, save the best column displacement field
                prediction[r:r+brow,c:c+bcol]= Rbest
                
                
    mvf = -mvf # For compatibility with standards
                               
    return mvf, prediction



def displayMVF(I,mvf,subsamp):
    
    row,col = np.shape(I)
    
    # Subsample dense MVF 
    vectRow = mvf[0:row:subsamp,0:col:subsamp,0]
    vectCol = mvf[0:row:subsamp,0:col:subsamp,1]
    
    # Determine vector origins
    X,Y = np.meshgrid(np.arange(0,col,subsamp),np.arange(0,row,subsamp))
    X = X + np.floor(subsamp/2.)
    Y = Y + np.floor(subsamp/2.)
    
    # Show image
    plt.figure()
    plt.imshow(I,cmap='gray')

    plt.quiver(X, Y, vectCol,-vectRow)

    
    return


def computeDerivatives(im1, im2):
    kernel1D = np.expand_dims((1./12.)*np.array([-1., 8., 0., -8., 1.]),axis=1).T
    fx = convolve2d(im1,kernel1D,'same')
    fy = convolve2d(im1,kernel1D.T,'same')
    ft = convolve2d(im1, 0.25*np.ones((2,2)),'same') + convolve2d(im2, -0.25*np.ones((2,2)),'same');
    fx =-fx;
    fy=-fy;

    return fx, fy, ft



def HS(im1, im2, uInitial, vInitial, alpha=1, ite=100,blur = 1.):
    """
    Horn-Schunck optical flow method.
    Horn, B.K.P., and Schunck, B.G., Determining Optical Flow, AI(17), No. 1-3, August 1981, pp. 185-203 http://dspace.mit.edu/handle/1721.1/6337
    
    [u, v] = HS(im1, im2, alpha, ite, uInitial, vInitial, displayFlow)
    
    Parameters
    ----------
    im1, im2 : np.array
        Two subsequent frames or images.
    uInitial, vInitial: np.array
        Initial values for the (u,v) flow. If well initialized, the flow would converge faster and hence would need less iterations.
    alpha : scalar
        A parameter that reflects the influence of the smoothness term.
    ite : int
        Number of iterations.

    Returns
    -------
    u,v : np.array
        Optical flow.
    """
    
    im1 = gaussian_filter(im1,blur)
    im2 = gaussian_filter(im2,blur)
    
    # Set initial value for the flow vectors
    u = uInitial.copy()
    v = vInitial.copy()
    
    # Estimate spatiotemporal derivatives
    fx, fy, ft = computeDerivatives(im1, im2)
    
    # Averaging kernel
    kernel_1=np.array([[1./12., 1./6., 1./12.],[1./6., 0., 1./6.],[1./12., 1./6., 1./12.]])
    
    for i in range(ite):
        # Compute local averages of the flow vectors using kernel_1
        uAvg=convolve2d(u,kernel_1,'same')
        vAvg=convolve2d(v,kernel_1,'same')
        
        # Compute flow vectors constrained by its local average and the optical flow constraints
        u = uAvg-fx*((uAvg*fx+vAvg*fy+ft)/(alpha**2+fx**2+fy**2))
        v = vAvg-fy*((uAvg*fx+vAvg*fy+ft)/(alpha**2+fx**2+fy**2))
        
    u[np.isnan(u)]= 0
    v[np.isnan(v)]= 0
    return u, v