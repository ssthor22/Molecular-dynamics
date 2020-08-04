import numpy as np
import matplotlib.pyplot as plt
import time
import random
import scipy.io as sio
import pickle

# Code written by Seth Thor
# Simulate hard sphere motion in box with periodic boundaries

# LATEST WORKING PBC HS
# Run dilute limit, phi = 0.01, L = 30*r
# Run intermediate, phi ~ 0.2,	L = 11*r 
# Run intermediate with large number of steps, phi ~ 0.2, L = 11*r
# 	- want to run Cv to get long-time D

# Output: MSD vs time, Cv vs t, PCF

# N = 64, phi = 0.01 

#===========================================================================
# EXP 1: L = 30*r
#===========================================================================

# define global variables

# N, number of particles
# T0, temperature in kelvin
# AMU, atomic mass unit in kg
# M1, mass of each Neon aton in amu
# M2, mass of each Argon atom in amu

N = 64;
T0 = 20;
AMU = 1.66054e-27;
M1 = 20.18*AMU;
M2 = 39.95*AMU;
r1 = 38e-12;
r2 = 71e-12;
#L = 1.5e-9;
kB = 1.38e-23;

M = M2
r = r2

# simulation time step
dt_god = 1E-14

# --------------------------------
# Periodic Box conditions

# Ri is a vector of length 3 containing
# x, y, z coordinates of ith atom in that order.
# function modifies Ri to place atom i into
# a periodic boundary box of length L centered about origin
# the box contains coordinates of [-L/2, L/2) and wraps around

def PutInBox(Ri,L):
    neg_box_edge = -L/2.0
    pos_box_edge = L/2.0
    
    #Check which coordinate needs to be remapped (any coordinate outside box)
    coord2remap = []
    for i in range( len(Ri) ):
        if Ri[i] > pos_box_edge or Ri[i] < neg_box_edge:
            coord2remap.append(i)
        else: pass
    
    if len(coord2remap) == 0:
        Ri = Ri # Particle already in box
    else:
    
        for i in coord2remap:
            if Ri[i] < 0: box_edge = neg_box_edge
            else: box_edge = pos_box_edge
        
            # Compute fraction of half-box length that particle is away from box
            # Floor the fraction.
            f = (Ri[i]-box_edge)/box_edge # fraction of half-box lengths where particle actually is
            floor = np.floor(f) # integer # of half-box lengths
            f_floored = f - floor # This will give fraction of half-box length b/t 0-1 where particle will be remapped to
            
            # Remap coordinate to be in the box
            Ri[i] = box_edge*f_floored + (-box_edge) 
        
    return Ri

# displacement function finds shortest distance vector
# from Rj to Ri on a PBC, by fixing Rj and varying Ri
# this function preserves the direction, vector points Ri<-Rj

def Displacement(Ri,Rj,L):
    
    dR = [0,0,0]
    for k in range( len(Ri) ):
        first = Ri[k] - Rj[k]
        second = Ri[k] - (Rj[k] - L)
        third = Ri[k] - (Rj[k] + L)
        
        min_disp = np.min( np.abs([first, second, third]) )
        
        if min_disp == np.abs(first):
            dR[k] = first
        elif min_disp == np.abs(second):
            dR[k] = second
        elif min_disp == np.abs(third):
            dR[k] = third

    dR = np.array(dR)        
    return dR

# calculates the distance between two atoms in PBC
# using the displacement funtion 
def Distance(Ri,Rj):
	dR = Rj - Ri;
	d = sum(dR**2)**0.5;
	return d;

# -----------------------------------------------
# Visualization aids

# writes out coordinates to a file readable by VMD
def VMDOut(coordList, R_alltimes, V_alltimes, L_i):

	atom_names = {1:'Ne ', 2:'Ar '};
	numPtcls=len(coordList[0]);
	outFile=open('HS_Trajectory_'+str(L_i)+'.xyz','w');
	for coord in coordList:
		for i in range(0,len(coord)):
			if i % 64 == 0: outFile.write("64\ncomment line\n")
			else: pass

			name = atom_names[1];
			if i == 0:
				name = atom_names[2];
			outFile.write(name+str(coord[i][0])+" "+str(coord[i][1])+" "+str(coord[i][2])+"\n");
	outFile.close()

	sio.savemat('HS_Velocities_'+str(L_i)+'.mat', mdict={'V_alltimes':V_alltimes})
	sio.savemat('HS_Positions_'+str(L_i)+'.mat', mdict={'R_alltimes':R_alltimes})

	return None;

# -----------------------------------------------
# Position and velocity initializers

# initializes the positions of the atoms in a cubic box
# will automatically split the atoms by half into two species
# separated by an optional split distance, default at zero
# returns an array of positions but with uninitialized species tags (1: Ne, 2:Ar)
# param: (int) N, number of atoms to initialize, will yell if not a perfect cube
# param: (float) L, length of one side of the cube
# param: (float) split = 0.0, distance of the x-split between species
# return: (array) [x, y, z] array of coordinates
def InitPositionCubic(N, L, split = 0.0):
	position = np.zeros((N,3)) + 0.0;
	Ncube = 1;
	while(N > (Ncube*Ncube*Ncube)):
		Ncube += 1;
	if(Ncube**3 != N):
		print("CubicInit Warning: Your particle number",N, \
			"is not a perfect cube; this may result " \
			"in a lousy initialization");
	width = float(L)/2 - split;
	length = float(L) - split;
	rw = 2*width/Ncube;
	rl = length/Ncube;
	roffset = float(L)/2 - split/2 - rl/2;
	negoffset = float(L)/2 - split/2 - rw/2;
	posoffset = float(L)/2 - 1.5*split - rw/2;
	added = 0;
	for x in range(0, Ncube):
		for y in range(0, Ncube):
			for z in range(0, Ncube):
				if(added < N/2):
					position[added, 0] = rw*x - negoffset;
					position[added, 1] = rl*y - roffset;
					position[added, 2] = rl*z - roffset;
					added += 1;
				elif(added >= N/2 and added < N):
					position[added, 0] = rw*x - posoffset;
					position[added, 1] = rl*y - roffset;
					position[added, 2] = rl*z - roffset;
					added += 1;
	return position;

# initializes the positions of the atoms in a cubic box
# will automatically split the atoms by half into two species
# optional toogle to initial pulse on left side of box
# param: (int) N, number of atoms to initialize
# param: (float) T0, syatem temperature
# param: (list) M = [1.0, 1.0], masses of the atom species
# param: (bool) pulse = False, toggles between usual and inital pulse
# return: (arrays) velocity, array of atom velocities
def InitVelocity(N, T0, mass = [1.0, 1.0], pulse = False):
	np.random.seed(1);
	velocity = np.random.normal(scale = (kB*T0/mass[0])**0.5, size = (N, 3));
	return velocity;

# ----------------------------------------------------------
# Hard sphere functions

# test if two atoms i and j will collide 
# param: (int) i, index of upstream atom
# param: (int) j, index of downlist atom (i < j)
# param: (float array) R, list of atom positions
# param: (float array) V, list of atom velocities
# return: (float) time till collision if True, INF otherwise
def collideTime(i, j, R, V):
#	rij = Displacement(R[i], R[j],L);
	rij = R[j] - R[i]
	vij = V[j] - V[i];
	b = np.dot(rij,vij)
	if b < 0:
		a = np.dot(vij, vij);
		c = np.dot(rij, rij) - (2*r)**2;
		d = b**2 - a*c;
		if d > 0:
			q = (-b + np.sqrt(d) )
			tij = min( q/a, c/q );
			if tij < 0: tij = max(q/a, c/q)
#			print(a, b, c, d, tij)
		else: tij = np.inf
	else: tij = np.inf

	return tij;

def UpdateV(i,j,R,V):
#	rij = Displacement(R[i],R[j],L)
	rij = R[j] - R[i]
	vij = V[j] - V[i]
	b = np.dot(rij,vij)
	dv = (b*rij)/np.dot(rij,rij)
	
	nV = V.copy()
	nV[i] = nV[i] + dv
	nV[j] = nV[j] - dv
	
	return nV

# i_collide: Particle i that already collided
# j_collide: Particle j that already collided
def UpdateCollisionPartners(nR,nV):
	
	ni_collide_times = np.zeros(shape=(N))
	ni_collide_times[N-1] = np.inf
	ni_collide_partners = np.zeros(shape=(N))
	ni_collide_partners[N-1] = np.inf

	for i in range(N-1):
		collide_times = []
		j_indices = []
		for j in range(i+1,N):
			tij = collideTime(i,j,nR,nV)
			collide_times.append(tij)
			j_indices.append(j)
		
#		print collide_times
		tij = np.min(collide_times)
		j_index = j_indices[ collide_times.index(tij) ]
		ni_collide_times[i] = tij
		if np.isinf(tij): ni_collide_partners[i] = np.inf
		else: ni_collide_partners[i] = j_index
		
	return ni_collide_partners, ni_collide_times

def ComputeTotalPandE(V):
	P = 0.
	totalKE = 0.
	for i in range(N):
		P += M*np.linalg.norm(V[i])
		vi_sq = np.dot(V[i],V[i])
		totalKE = 0.5*M*vi_sq

	return P, totalKE

def hardsphere_sim(steps, L, split=0.0):

	# initalize positions and velocities
	R = InitPositionCubic(N, L, split);
	R_alltimes = np.zeros((steps,N,3))
	V = InitVelocity(N, T0, mass = [M, M]);
	V_alltimes = np.zeros((steps, N, 3));
	tags = np.array([[np.inf, np.inf] for i in range(N)]);

	coordList = np.zeros((steps, N, 3));
	P = np.zeros(steps);
	E = np.zeros(steps);

	# Identify 1st set of collision partners and collision times
	i_collide_times = np.zeros(shape=(N))
	i_collide_times[N-1] = np.inf
	i_collide_partners = np.zeros(shape=(N))
	i_collide_partners[N-1] = np.inf
	for i in range(N-1):
		collide_times = []
		j_indices = []
		for j in range(i+1,N):
			tij = collideTime(i,j,R,V)
			collide_times.append(tij)
			j_indices.append(j)
		
#		print collide_times
		tij = np.min(collide_times)
		j_index = j_indices[ collide_times.index(tij) ]
		i_collide_times[i] = tij
		if np.isinf(tij): i_collide_partners[i] = np.inf
		else: i_collide_partners[i] = j_index

	t = 0.
	dtij = min( i_collide_times ) # identify first collision time
	sim_tij = t + dtij
#	raw_input()
	simtime = 0.
	simtime_steps = []
	for n in range(steps):
#		print i_collide_partners
#		print i_collide_times
		dt = dt_god
		t = n*dt
		simtime_steps.append(t)
		coordList[n] = R
		P[n],E[n] = ComputeTotalPandE(V)
		V_alltimes[n] = V.copy()
		R_alltimes[n] = R.copy()
		if t < sim_tij: # no collision occurs, just update positions
			nR = R + V*dt
			for particle in range(N):
				PutInBox( nR[particle],L )
			nV = V.copy()

			R = nR.copy()
			V = nV.copy()
		else: # run collision dynamics and find new collision times/partners
			while t >= sim_tij:
				print "Collision detected b/t time step"
				print "n = "+str(n)+" | t = "+str(t)+" | tij = "+str(sim_tij)
				t = simtime_steps[n-1]				# go back to previous time step
				dt = sim_tij - t				# compute dt to time of collision
				t = sim_tij					# We are now at time of collision
				nR = R + V*dt					# progress positions to time of collision
				for particle in range(N):
					PutInBox( nR[particle],L )
				i = i_collide_times.tolist().index(dtij)
				j = i_collide_partners[i]
				nV = UpdateV(i,j,R,V)				# update particle velocities after collision

				# Update collision partners and times
				ni_collide_partners, ni_collide_times = UpdateCollisionPartners(nR,nV)

				# Update variables for next time step
				i_collide_times = ni_collide_times.copy()
				i_collide_partners = ni_collide_partners.copy()
				dtij = min( i_collide_times ) 			# identify next collision time interval
				sim_tij_old = sim_tij				# record old collision time that we are currently at
				sim_tij = t + dtij				# Next collision time
				t = simtime_steps[n]				# set simulation time to n*dt

				R = nR.copy()
				V = nV.copy()
	
			# Progress simulation from old collision time to next simulation time step
			dt = t - sim_tij_old
			nR = R + V*dt
			for particle in range(N):
				PutInBox( nR[particle],L )
			nV = V.copy()

			R = nR.copy()
			V = nV.copy()

	return coordList, P, E, V_alltimes, R_alltimes

def compute_D(coordList,steps,L_i):

	dt = dt_god	

	# compute D = < r_mean - ri >^2/t
	d_squared = []
	times = []
	R0 = coordList[0]
	MSDs = []
	for n in range(1,steps):
		R_t = coordList[n]
		for i in range(N):
			dr = R_t[i] - R0[i] 
			d_squared.append( np.dot(dr,dr) )

		MSDs.append( np.mean(d_squared) )
		times.append(n*dt)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(times,MSDs)
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('MSD')
	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.savefig('MSD_vs_t_loglog_'+str(L_i)+'.png',dpi=300)
	plt.clf()
	plt.close(fig)

	plt.figure()
	plt.plot(times,MSDs)
	plt.xlabel('Time')
	plt.ylabel('MSD')
	plt.savefig('MSD_vs_t_'+str(L_i)+'.png',dpi=300)
	plt.clf()
	plt.close(fig)

	pickle.dump( MSDs, open('MSDs_'+str(L_i)+'.p','wb') )
	pickle.dump( times, open('MSDs_times_'+str(L_i)+'.p','wb') )

	return

def ComputeCv(steps,L_i):
    
    V_alltimes = sio.loadmat('HS_Velocities_'+str(L_i)+'.mat')
    h = 1E-14
    C_v = []
    times = []
    for n in range(steps):
        V_all_at_0 = V_alltimes['V_alltimes'][0] # Velocity matrix at time 0
        V_all_at_t = V_alltimes['V_alltimes'][n] # Velocity matrix at time t_max
        summation = 0.0
        for i in range(N):
            summation += np.dot( V_all_at_0[i], V_all_at_t[i] )
        
        C_v.append( (1.0/N)*summation )
	times.append( h*n )
    
    # Integrate via trapezoidal rule to get D
    D = 0
    for n in range(0, steps-1):
        D += (h/2.0)*(C_v[n] + C_v[n+1])

    fig = plt.figure()
    plt.plot(times,C_v/C_v[0])
    plt.xlabel("Time")
    plt.ylabel(r"$C_v$", fontsize=18)
    plt.savefig('Cv_Ar_'+str(L_i)+'.png',dpi=300)
    plt.clf()
    plt.close(fig)

    pickle.dump( C_v, open('Cv_'+str(L_i)+'.p','wb') )
    pickle.dump( times, open('Cv_times_'+str(L_i)+'.p','wb') )

    return C_v, D

def PairCorrFn(R,nhis,L):
    # Computes the normalized pair correlation function for a configuration of particles
    # Initializes bins
    dist = np.zeros(nhis)

    # delg is based on the max length / number of bins
    delg = (L/2)*np.sqrt(3)/nhis

    # sum over particle pairs
    g = np.zeros((nhis))
    for i in range(0,len(R)-1):
        for j in range((i+1),len(R)):
            r=Distance(R[i],R[j])
            if r<L/2:
                ig=int(r/delg)
                g[ig]=g[ig]+2

    # particle normalization
    for l in range(0,len(g)):
        r=0.0
        vb=0.0
        nid = 0.0
        r=delg*(l+0.5)
        vb = ((l+1)**3-l**3)*delg**3
        nid = (4./3.*np.pi*vb*(N/L**3))
        g[l] = g[l]/(N*nid)
    for i in range(0,nhis):
        dist[i] = (delg/2.0) + i*delg

    return dist, g

def compute_PCF(R_alltimes,steps,L):

	nhis = 500
	PCF_dist = PairCorrFn(R_alltimes[0],nhis,L)[0]
	PCF = []
	print "Computing g(r) for L = "+str(L)
	for n in range( steps ):
		g = PairCorrFn(R_alltimes[n],nhis,L)[1]
		PCF.append(g)	
	
	PCF_avg = np.mean(PCF,axis=0)
	pickle.dump( PCF_avg, open('PCF_avg.p','wb') )
	pickle.dump( PCF_dist, open('PCF_dist.p','wb') )

	plt.figure()
	plt.plot(PCF_dist,PCF_avg)
	plt.xlabel('Distance')
	plt.ylabel('g(r)')
	plt.xlim([0,L/2])
	plt.savefig('PCF.png',dpi=300)
	plt.clf()

def main():

	# compute box size based on Vol Frac
#	phi = 0.7
#	L = ( ((4./3.)*N*np.pi*r**3 )/phi )**(1./3.)
#	L = 5E-10
#	print L
#	raw_input()

	f = open('log.log','wb')
	steps = 50000
	L_values = [30*r]
	for i in range( len(L_values) ):
		L = L_values[i]
		phi = ( ((4./3.)*N*np.pi*r**3)/L**3 ) # volume fraction
		print "L = "+str(L)+" | phi = "+str(phi)
		raw_input()
		start = time.time();
		coordList, P, E, V_alltimes, R_alltimes = hardsphere_sim(steps, L, split = 0.);
		end = time.time();
		print "HS sim time: "+str(end - start)

		# Save coordinate list and particle velocities
		VMDOut(coordList, R_alltimes, V_alltimes, i);

		# Plot momentum
		np.savetxt('HS_momentum_'+str(i)+'.dat', P);
		fig = plt.figure()
		plt.plot(P)
		plt.xlabel('Step')
		plt.ylabel('Total Momentum')
#		plt.ylim([0,10E-22])
		plt.savefig('TotalP'+str(i)+'.png',dpi=300)
		plt.clf()
		plt.close(fig)

		# Plot momentum
		np.savetxt('HS_energy_'+str(i)+'.dat', E);
		fig = plt.figure()
		plt.plot(E)
		plt.xlabel('Step')
		plt.ylabel('Total E')
#		plt.ylim([0,10E-22])
		plt.savefig('TotalE'+str(i)+'.png',dpi=300)
		plt.clf()
		plt.close(fig)

		# Compute MSD vs Time data
		print "Computing MSD vs Time data..."
		compute_D(coordList,steps,i)

		# D from velocity correlation
		print "Computing Cv data..."
		C_v, D = ComputeCv(steps,i)
		print "Diffusion Coeff: "+str(D)

#		PAIR CORRELATION STUFF
		compute_PCF(R_alltimes,steps,L)

		# Output comments to log file
		f.write("====================================================================\n")
		f.write(str(i)+" | L = "+str(L)+" | phi = "+str(phi)+" | D(Cv) = "+str(D)+"\n")

	f.close()

main()
		
		
		



