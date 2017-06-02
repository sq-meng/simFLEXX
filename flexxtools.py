import numpy as np
from scipy import interpolate
import configparser
import json
import csv
import time
#
# The core of MultiFLEXX tools, handles coordinate system conversions.
# Please use right-hand convention when defining scattering plane with hkl1 and hkl2. The opposite case has not been
# thoroughly tested.
#

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class UBmatrix(object):

    def calculateBinL(self):
        latparam = self.conf.sample['latparam']
        a, b, c, alphadeg, betadeg, gammadeg = latparam

        cosalph = np.cos(alphadeg * np.pi / 180)

        cosbet = np.cos(betadeg * np.pi / 180)

        cosgam = np.cos(gammadeg * np.pi / 180)
        singam = np.sin(gammadeg * np.pi / 180)
        tangam = np.tan(gammadeg * np.pi / 180)

        ax = a
        ay = 0
        az = 0
        aL = np.array([[ax], [ay], [az]])

        bx = b * cosgam
        by = b * singam
        bz = 0
        bL = np.array([[bx], [by], [bz]])

        cx = c * cosbet
        cy = c * (cosalph / singam - cosbet / tangam)
        cz = c * np.sqrt(1 - (cx ** 2 + cy ** 2) / c ** 2)
        cL = np.array([[cx], [cy], [cz]])

        BinL = np.concatenate((aL, bL, cL), axis=1)

        return BinL

    def calculateRinL(self):
        RinL = (2 * np.pi * np.linalg.inv(self.__BinL)).T
        return RinL

    def calculateSinL(self):
        G1L = np.dot(self.__RinL, self.conf.alignment['hkl1'])
        G2L = np.dot(self.__RinL, self.conf.alignment['hkl2'])

        G1xG2 = np.cross(G1L, G2L)
        SZinL = G1xG2 / np.linalg.norm(G1xG2)
        SXinL = G1L / np.linalg.norm(G1L)
        SYinL = np.cross(SZinL, SXinL)

        SinL = np.array([SXinL, SYinL, SZinL]).T

        return SinL

    def calculatePinL(self):
        PXinL = np.dot(self.__RinL, self.conf.plot['x'])
        PYinL = np.dot(self.__RinL, self.conf.plot['y'])
        PZinL = np.cross(PXinL, PYinL)  # for completeness

        PinL = np.array([PXinL, PYinL, PZinL]).T
        return PinL

    def convert(self, coord, sys):
        if sys == 'RS':
            L = np.dot(self.__RinL, coord)
            S = np.dot(self.__LinS, L)
            return S
        elif sys == 'SR':
            L = np.dot(self.__SinL, coord)
            R = np.dot(self.__LinR, L)
            return R
        elif sys == 'RP':
            L = np.dot(self.__RinL, coord)
            P = np.dot(self.__LinP, L)
            return P
        elif sys == 'SP':
            L = np.dot(self.__SinL, coord)
            P = np.dot(self.__LinP, L)
            return P
        elif sys == 'PR':
            L = np.dot(self.__PinL, coord)
            R = np.dot(self.__LinR, L)
            return R

    def find_kS(self, QR, ch):
        ki = self.conf.instrument['ki']
        kf = self.conf.instrument['kf'][ch - 1]
        QS = self.convert(QR, 'RS')
        QSL = np.linalg.norm(QS)
        alpha, beta, gamma = find_triangle(QSL, ki, kf)
        kiS = -rotZ(QS, gamma) / QSL * ki
        kfS = rotZ(QS, -beta) / QSL * kf
        return kiS, kfS

    def __init__(self, conf):
        self.conf = conf

        self.__BinL = self.calculateBinL()
        self.__RinL = self.calculateRinL()
        self.__LinR = np.linalg.inv(self.__RinL)
        self.__SinL = self.calculateSinL()
        self.__LinS = np.linalg.inv(self.__SinL)
        self.__PinL = self.calculatePinL()
        self.__LinP = np.linalg.inv(self.__PinL)


# noinspection PyDictCreation
def loadConfig(config_parser):
    conf_sample = {}
    conf_sample['latparam'] = json.loads(config_parser.get('Sample', 'latparam'))

    alignment = {}
    alignment['hkl1'] = json.loads(config_parser.get('Alignment', 'hkl1'))
    alignment['hkl2'] = json.loads(config_parser.get('Alignment', 'hkl2'))

    instrument = {}
    instrument['ki'] = config_parser.getfloat('Instrument', 'ki')
    instrument['ei'] = (instrument['ki'] / 0.6942) ** 2
    instrument['A4'] = config_parser.getfloat('Instrument', 'A4')
    instrument['ef'] = json.loads(config_parser.get('Instrument', 'ef'))
    instrument['kf'] = list(map(lambda x: 0.6942 * np.sqrt(x), instrument['ef']))
    instrument['channels'] = config_parser.getint('Instrument', 'channels')
    instrument['choffset'] = config_parser.getfloat('Instrument', 'channel_offset')

    conf_scan = {}
    conf_scan['A3start'] = config_parser.getfloat('Scan', 'A3start')
    conf_scan['A3end'] = config_parser.getfloat('Scan', 'A3end')

    conf_plot = {}
    conf_plot['x'] = json.loads(config_parser.get('Plot', 'x'))
    conf_plot['y'] = json.loads(config_parser.get('Plot', 'y'))
    conf_plot['xlabel'] = config_parser.get('Plot', 'xlabel')
    conf_plot['ylabel'] = config_parser.get('Plot', 'ylabel')

    conf_horizontal_magnet = {}
    conf_horizontal_magnet['magnet_ident'] = config_parser.get('Horizontal_magnet', 'magnet_ident')
    conf_horizontal_magnet['north_along'] = json.loads(config_parser.get('Horizontal_magnet', 'north_along'))
    conf_horizontal_magnet['sample_stick_rotation'] = \
        config_parser.getfloat('Horizontal_magnet', 'sample_stick_rotation')
    return conf_sample, alignment, instrument, conf_scan, conf_plot, conf_horizontal_magnet


class config(object):
    def __init__(self, config_file):
        config_parser = configparser.ConfigParser()
        config_parser.read(config_file)
        sample, alignment, instrument, scan, plot, horizontal_magnet = loadConfig(config_parser)
        self.sample = sample
        self.alignment = alignment
        self.instrument = instrument
        self.scan = scan
        self.plot = plot
        self.horizontal_magnet = horizontal_magnet
        self.transmission = []
        self.loadMagnetTransmission()

    def loadMagnetTransmission(self):
        ident = self.horizontal_magnet['magnet_ident']
        if ident == 'none':
            self.horizontal_magnet['exist'] = False
            return
        else:
            try:
                file = open(ident + '.csv')
                reader = csv.DictReader(file)
                theta = []
                transmission = []
                for row in reader:
                    theta.append(float(row['theta']))
                    transmission.append(float(row['transmission']))
                self.horizontal_magnet['exist'] = True
                print('Loaded HM transmission profile with ' + str(len(theta)) + ' entries.')
                self.transmission = interpolate.interp1d(np.radians(np.array(theta)), np.array(transmission))
            except FileNotFoundError:
                print('Error loading magnet transmission profile, assuming NO horizontal magnet.')
                self.horizontal_magnet['exist'] = False
            finally:
                pass


# noinspection PyPep8Naming
def angleToPlot(core, ki, kf, A3, A4, hm=False, ssr=0.0):
    kiS = ki * np.array([-1, 0, 0])
    kfS = kf * rotZ(np.array([-1, 0, 0]), np.radians(A4))
    QS = rotZ((kfS - kiS), - np.radians(A3))
    QP = core.convert(QS, 'SP')
    if not hm:
        return QP
    else:
        if not core.conf.horizontal_magnet['exist']:
            print('HM shadow requested but not properly initialized.')
            return QP, 1.0
        else:
            magnet_orientation_R = core.conf.horizontal_magnet['north_along']
            magnet_orientation_S = core.convert(magnet_orientation_R, 'RS')
            ki_azimuth = np.remainder(azimuthS(magnet_orientation_S, kiS) + np.pi - np.radians(A3) + np.radians(ssr),
                                      2 * np.pi)
            kf_azimuth = np.remainder(azimuthS(magnet_orientation_S, kfS) - np.radians(A3) + np.radians(ssr),
                                      2 * np.pi)
            ki_transmission = core.conf.transmission(ki_azimuth)
            kf_transmission = core.conf.transmission(kf_azimuth)
            return QP, ki_transmission * kf_transmission


def rotZ(coord, angle):
    #
    # Rotates provided vector around [0, 0, 1] for given degrees counterclockwise.
    #
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])

    return np.dot(matrix, coord)


def azimuthS(k1, k2):
    #
    # Calculates how much k1 need to be rotated counterclockwise to reack k2.
    #
    if np.cross(k1, k2)[2] >= 0:
        return np.arccos(np.dot(k1, k2) / np.linalg.norm(k1) / np.linalg.norm(k2))
    else:
        return -1 * np.arccos(np.dot(k1, k2) / np.linalg.norm(k1) / np.linalg.norm(k2))


def find_triangle(a, b, c):
    #
    # Receives side lengths of triangle and calculate corresponding angles.
    #
    aa = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    bb = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    cc = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    if aa > 1 or aa < -1 or bb > 1 or bb < -1 or cc > 1 or cc < -1:
        raise ValueError('Scattering triangle cannot close.')
    alpha = np.arccos(aa)
    beta = np.arccos(bb)
    gamma = np.arccos(cc)

    return alpha, beta, gamma
