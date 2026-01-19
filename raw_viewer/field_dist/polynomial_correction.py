import cupy as cp
import numpy as np

class PolynomialCorrection:
    '''
    Corrections will be performed as follows.
    Track width will first be corrected as a function of track angle. This is needed because of different diffusion coefficients in xy vs z,
    and because of amplifier shaping time affecting z width. Width may also be corrected as funciton of time, if desired.
    The correction to width is of form width => width + sum_(i,j,k) a_ij*(angle^i)*(width^j)*(time^k)

    Points are also translated relative to a beam spot center and then converted to cylindrical cordinates,
    and then each points is corrected like this:
    r => r + sum_(ijklm) a_ijklmn r^i sin(theta)^j cos(theta)^k z^l corrected_width^m time^n
    and similarly for theta and z. Typically only one of sin or cos(theta) needs to be use, but the option to use both is 
    included for completeness, as can any z depenence. Parameterization is in terms of sin and cos to ensure smoothness at theta=0/2pi.
    And typically z->z will be a fine mapping unless there is a huge field distorition
    because we typically operate the detector in a region where drift speed is nearly independent of field strength.
    The points are then translated back relative to the beam spot center.

    Typical usage:
    Call set data to transfer data to GPU
    Set up fit parameters
    Call

    '''

    def __init__(self, gpu_id_to_use):
        self.gpu_id_to_use = gpu_id_to_use #don't change after calling set_data for the first time

        #inputs
        self.uncorrected_points = None
        self.uncorrected_width = None
        self.uncorrected_angles = None
        self.times = None

        #beam spot center
        self.beam_spot_center = (0,0)

        #correction parameters. Defaults to no correction applied.
        self.width_ijk = [] 
        self.width_parameters = []

        self.r_ijklmn = []
        self.r_parameters = []
        self.theta_ijklmn = []
        self.theta_parameters = []
        self.z_ijklmn = []
        self.z_parameters = []

        self.width_scale = 3
        self.r_scale = 30
        self.z_scale = 50
        self.time_scale = 0.1

        self.r_max = 61 #mm, radius of field cage

        #intermediate variables
        self.uncorrected_r, self.uncorrected_theta, self.uncorrected_z = None, None, None #holds r, theta, phi pairs after shifting over by beamspot
        self.corrected_r, self.corrected_theta = None, None #holds r, theta, phi pairs after applyng polynomial correction
        
        self.powers_dict = {} #dictionary of width, z, r, etc raised to required powers. Saved to avoid uneeded recalcultations

        #outputs
        #these member variables will hold the mapped points once set_data and apply_correction have been called
        self.corrected_xyz = None
        self.corrected_width = None
    
    def set_data(self, points, widths, track_angles, times):
        '''
        Saves inputs and transfers the data to the GPU. Allocate space for intermediate variables and outputs.
        '''
        with cp.cuda.Device(self.gpu_id_to_use):
            self.points = cp.array(points)
            self.uncorrected_width = cp.array(widths)
            self.uncorrected_angles = cp.array(track_angles)
            self.times = cp.array(times)
            #allocate space for corrected points
            self.corrected_xyz = cp.zeros(cp.shape(points))
            #reset previously calculted powers of parameters
            self.powers_dict = {}

    def get_power(self, parameter, power):
        with cp.cuda.Device(self.gpu_id_to_use):
            if (parameter, power) not in self.powers_dict:
                scale = 1
                if 'width' in parameter:
                    scale = self.width_scale
                elif parameter == 'times':
                    scale = self.time_scale
                elif '_r' in parameter:
                    scale = self.r_scale
                elif '_z' in parameter:
                    scale = self.z_scale
                self.powers_dict[(parameter, power)] = (self.__dict__[parameter]/scale)**power
            return self.powers_dict[(parameter,power)]


    def apply_width_correction(self):
        '''
        The correction to width is of form width => width + sum_(i,j,k) a_ij*(angle^i)*(width^j)*(time^k)
        Save result in corrected width member variable
        '''
        with cp.cuda.Device(self.gpu_id_to_use):
            self.corrected_width = cp.array(self.uncorrected_width, copy=True)
            for ijk, a_ijk in zip(self.width_ijk, self.width_parameters):
                i,j,k = ijk
                self.corrected_width += a_ijk*self.get_power('uncorrected_angles', i)*self.get_power('uncorrected_width', j)*self.get_power('times', k)
            #remove previously calculate powers of corrected width
            keys_to_remove = []
            for key in self.powers_dict.keys():
                if key[0] == 'corrected_width':
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.powers_dict[key]


    def convert_to_cylindrical_coords(self):
        '''
        Convert to cylindrical coordinates relative to beam spot center. This function should be called each time the beam spot center
        is changed.
        '''
        with cp.cuda.Device(self.gpu_id_to_use):
            xy = self.points[:,:2] - cp.array(self.beam_spot_center)
            self.uncorrected_z = self.points[:,2]
            self.uncorrected_theta = cp.arctan2(xy[:,1], xy[:,0])
            self.uncorrected_sin_theta = cp.sin(self.uncorrected_theta)
            self.uncorrected_cos_theta = cp.cos(self.uncorrected_theta)
            self.uncorrected_r = cp.sqrt(xy[:,0]*xy[:,0] + xy[:,1]*xy[:,1])
        #remove previously calculated powers of these parameters
        keys_to_remove = []
        for key in self.powers_dict.keys():
                if 'uncorrected_z' == key[0] or 'uncorrected_theta' == key[0] or 'uncorrected_sin_theta' == key[0] or \
                    'uncorrected_cos_theta' == key[0] or 'uncorrected_r' == key[0]:
                    keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.powers_dict[key]

    def apply_field_correction(self):
        '''
        Must first call set_data, apply_width_correction and convert_to_cylindrical_coords. 
        '''
        with cp.cuda.Device(self.gpu_id_to_use):
            #r => r + sum_(ijklm) a_ijklmn r^i sin(theta)^j cos(theta)^k z^l corrected_width^m time^n
            self.corrected_r = cp.array(self.uncorrected_r, copy=True)
            for ijklmn, a_ijklmn in zip(self.r_ijklmn, self.r_parameters):
                i,j,k,l,m,n = ijklmn
                self.corrected_r += a_ijklmn*self.get_power('uncorrected_r', i)*self.get_power('uncorrected_sin_theta', j) \
                                    *self.get_power('uncorrected_cos_theta', k)*self.get_power('uncorrected_z', l)\
                                    *self.get_power('corrected_width', m)*self.get_power('times',n)
                
            self.corrected_r[self.corrected_r>self.r_max] = self.r_max
            self.corrected_r[self.corrected_r<-self.r_max] = -self.r_max
            #theta
            self.corrected_theta = cp.array(self.uncorrected_theta, copy=True)
            for ijklmn, a_ijklmn in zip(self.theta_ijklmn, self.theta_parameters):
                i,j,k,l,m,n = ijklmn
                self.corrected_theta += a_ijklmn*self.get_power('uncorrected_r', i)*self.get_power('uncorrected_sin_theta', j) \
                                    *self.get_power('uncorrected_cos_theta', k)*self.get_power('uncorrected_z', l)\
                                    *self.get_power('corrected_width', m)*self.get_power('times',n)
            #z
            self.corrected_xyz[:,2] = cp.array(self.uncorrected_z, copy=True)
            for ijklmn, a_ijklmn in zip(self.z_ijklmn, self.z_parameters):
                i,j,k,l,m,n = ijklmn
                self.corrected_xyz[:,2] += a_ijklmn*self.get_power('uncorrected_r', i)*self.get_power('uncorrected_sin_theta', j) \
                                    *self.get_power('uncorrected_cos_theta', k)*self.get_power('uncorrected_z', l)\
                                    *self.get_power('corrected_width', m)*self.get_power('times',n)
                
            #convert back to cartesian coordinates
            self.corrected_xyz[:,0] = self.beam_spot_center[0] + self.corrected_r*cp.cos(self.corrected_theta)
            self.corrected_xyz[:,1] = self.beam_spot_center[1] + self.corrected_r*cp.sin(self.corrected_theta)