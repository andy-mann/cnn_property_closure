import numpy as np
import h5py

dir = '/Users/andrew/Dropbox (GaTech)/code/class/materials_informatics/inputs/expand/delta.npy'


fp = '/Users/andrew/Dropbox (GaTech)/2PS_Property_Data/2PS_train.h5'
dat = h5py.File(fp)
stats = dat['2PS']
shift_stats = np.fft.ifftshift(stats,axes=(1,2,3))

data = np.load(dir)
data = np.fft.ifftshift(data,axes=(1,2,3))


def twoptcombo(x1, x2, alpha):
    '''
    x2 should be the boundary point
    '''
    return (1-alpha) * x1.real + (alpha)*x2.real
    #return (.1) * x1.real + (alpha)*x2.real


def extrapolate(p1, p2, step_size=.1,lim = 50):
    fp1 = np.fft.fftn(p1,axes=(0,1,2))
    fp2 = np.fft.fftn(p2,axes=(0,1,2))

    candidate = p2
    alpha = 1
    count = 0
    valid = True
    while valid and count <= lim:
        #print(count)
        new_point = twoptcombo(p1, p2, alpha)
        if check_valid(new_point):
            valid = True
            candidate = new_point
            count += 1
        else:
            valid = False
        alpha += step_size
        
    print(count)
    return candidate, count

'''
def check_valid(candidate):
    
    #stats must have max value at 0,0,0
    #must remove idx dimension
    
    fstat = np.fft.fftn(candidate, axes=(0,1,2))
    if np.unravel_index(np.argmax(candidate), candidate.shape) == (0,0,0):
        #print('(0,0,0) component is largest')
        if abs(np.max(candidate.imag)) < 1e-9:
            #print('strictly real')
            if np.min(candidate.real) > -1e-9:
                #print('stricly non-negative')
                if abs(np.max(fstat.imag)) < 1e-9:
                    #print('fft is real')
                    if np.min(fstat.real) > -1e-9:
                        #print('fft is strictly non-negative')
                        if fstat.min().real > -1e-12:
                            #print('This is a valid set of 2-point statistics!')
                            return True
                        else:
                            print('invalid - condition 6')
                            return False
                    else:
                        print('invalid - condition 5')
                        return False
                else:
                    print('invalid - condition 4')
                    return False
            else:
                print('invalid - condition 3')
                return False
        else:
            print('invalid - condition 2')
            return False
    else:
        print('invalid - condition 1')
        return False
'''

def check_valid(candidate):
    #shift = np.fft.ifftshift(candidate,axes=(0,1,2))
    r = np.fft.ifftn(candidate,axes=(0,1,2))

    if np.unravel_index(np.argmax(r), candidate.shape) == (0,0,0):
        if np.all(np.greater(r,0)):
            #print('strictly positive')
            if np.all(np.less(r,1)):
                #print('stricly less than 1')
                if np.min(candidate.real) > -1e-9:
                    #return True
                    if np.min(candidate.imag) > -1e-9:
                        return True 
                    else:
                        print('condition 4')
                        return False
                else:
                    print('condition 3')
                    return False
            else:
                print('invalid - condition 2')
                return False
        else:
            #print('invalid - condition 1')
            return False
    else:
        return False

'''
p = np.zeros((14*14, 31,31,31))
idx=0
for i in range(14):
    for j in range(14):
        #print(f'{i} and {j}')
        p[idx] = extrapolate(data[i], data[j])
        idx = idx + 1
        #print(idx)

pr = np.fft.ifftn(p,axes=(1,2,3))
pshift = np.fft.fftshift(pr, axes=(1,2,3))
fp = '/Users/andrew/Dropbox (GaTech)/code/class/materials_informatics/inputs/expand/extrap_fourier_stats.npy'
np.save(fp, pshift)
'''
arr = []
for i in range(len(shift_stats)):
    stat, count  = extrapolate(shift_stats[i],data[1])
    if count > 2:
        arr.append(stat)

stats = np.array(arr)

pr = np.fft.ifftn(stats,axes=(1,2,3))
pshift = np.fft.fftshift(pr, axes=(1,2,3))
fp = '/Users/andrew/Dropbox (GaTech)/code/class/materials_informatics/inputs/expand/delta_extrap_2.npy'
np.save(fp, pshift)
