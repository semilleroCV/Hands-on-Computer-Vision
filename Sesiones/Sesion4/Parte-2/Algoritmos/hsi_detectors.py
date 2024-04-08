#
# hsi_detectors.py
# 
# Copyright 2018 - Taylor Glenn - tcg@precisionsilver.com
#

import numpy as np
import math

def topix(x):
	return x.reshape( (x.shape[0]*x.shape[1],x.shape[2]) )

def toimg(pix,x,n_dim=None):
	if n_dim is None:
		n_dim = len(x.shape)

	return pix.reshape(x.shape[:n_dim])


def smf_detector(x,tgt_sig,mu=None,siginv=None):

	pix = topix(x)
	n_pix,n_band = pix.shape

	if mu is None:
		mu = pix.mean(axis=0)
	if siginv is None:
		sig = np.cov(pix.T)
		siginv = np.linalg.inv(sig)

	s = tgt_sig - mu
	z = pix - mu
	f = s.dot(siginv) / math.sqrt(s.dot(siginv.dot(s)))
	smf_data = z.dot(f)
	
	return toimg(smf_data,x,2)
	
def ace_detector(x,tgt_sig,mu=None,siginv=None):

	pix = topix(x)
	n_pix,n_band = pix.shape

	if mu is None:
		mu = pix.mean(axis=0)
	if siginv is None:
		sig = np.cov(pix.T)
		siginv = np.linalg.inv(sig)

	s = tgt_sig - mu
	z = pix - mu
	st_siginv = s.dot(siginv)

	A = z.dot(st_siginv)
	B = np.sqrt(st_siginv.dot(s))
	C = np.sqrt(np.sum(z * z.dot(siginv.T), axis=1))
	ace_data = A/(B*C)

	return toimg(ace_data,x,2)