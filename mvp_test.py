import numpy as np

# for i, f in enumerate(self.flights):
#     x = f.position.x
#     y = f.position.y
#     vx, vy = f.components

num_ac = 2
t_look_ahead = 5
pzh = 2
margin = 1.05

x_array = np.array([[6.,2]])
y_array = np.array([[3.,6.]])

vx_array = np.array([[0.,np.sqrt(2.)]])
vy_array = np.array([[2.,np.sqrt(2.)]])

I = np.eye(len(x_array[0]))

dx = x_array - x_array.T
dy = y_array - y_array.T

dist = np.sqrt(dx*dx + dy*dy)

dvx = vx_array - vx_array.T
dvy = vy_array - vy_array.T

dv2 = dvx * dvx + dvy * dvy
dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
vrel = np.sqrt(dv2)

tcpa = -(dvx * dx + dvy * dy) / dv2 + 1e9 * I

# Calculate distance^2 at CPA (minimum distance^2)
dcpa2 = np.abs(dist * dist - tcpa * tcpa * dv2)
dcpa = np.sqrt(dcpa2)

d_heading = np.zeros((num_ac,1))

for i in range(num_ac):
    delta_vx = 0
    delta_vy = 0
    for j in range(num_ac):
        if (tcpa[i,j] < t_look_ahead and tcpa[i,j] > 0 and dcpa[i,j] < pzh * margin) or (dist[i,j] < pzh * margin):
            dcpa_x = dx[i,j] + dvx[i,j]*tcpa[i,j]
            dcpa_y = dy[i,j] + dvy[i,j]*tcpa[i,j]
            

            # Compute horizontal intrusion
            iH = pzh - dcpa[i,j]
            # print(dcpa[i,j])
            # print(dist[i,j])
            # Exception handlers for head-on conflicts
            # This is done to prevent division by zero in the next step
            if dcpa[i,j] <= 0.01:
                dcpa[i,j] = 0.01
                dcpa_x = dy[i,j] / dist[i,j] * dcpa[i,j]
                dcpa_y = -dx[i,j] / dist[i,j] * dcpa[i,j]
            
            # If intruder is outside the ownship PZ, then apply extra factor
            # to make sure that resolution does not graze IPZ
            if pzh < dist[i,j] and dcpa[i,j] < dist[i,j]:
                # Compute the resolution velocity vector in horizontal direction.
                # abs(tcpa) because it bcomes negative during intrusion.
                erratum = np.cos(np.arcsin(pzh / dist[i,j])-np.arcsin(dcpa[i,j] / dist[i,j]))
                dv1 = ((pzh / erratum - dcpa[i,j]) * dcpa_x) / (abs(tcpa[i,j]) * dcpa[i,j])
                dv2 = ((pzh / erratum - dcpa[i,j]) * dcpa_y) / (abs(tcpa[i,j]) * dcpa[i,j])
            else:
                dv1 = (iH * dcpa_x) / (abs(tcpa[i,j]) * dcpa[i,j])
                dv2 = (iH * dcpa_y) / (abs(tcpa[i,j]) * dcpa[i,j])
            
            delta_vx -= dv1
            # print('dvx',dv1)
            delta_vy -= dv2
            # print('dvy',dv2)

    new_vx = vx_array[0][i] + delta_vx
    print('new vx', new_vx)
    new_vy = vy_array[0][i] + delta_vy
    print('new vy', new_vy)
    # print(np.arctan2(0,1))
    # print(vx_array[0][i],vy_array[0][i])
    oldtrack = (np.arctan2(vx_array[0][i],vy_array[0][i])*180/np.pi) % 360
    print(oldtrack)
    newtrack = (np.arctan2(new_vx,new_vy)*180/np.pi) % 360
    print(newtrack)
    d_heading[i] = oldtrack-newtrack

print(d_heading)

##TODO IMPLEMENT THIS ALSO:

# def resumenav(self, conf, ownship, intruder):
#     '''
#         Decide for each aircraft in the conflict list whether the ASAS
#         should be followed or not, based on if the aircraft pairs passed
#         their CPA.
#     '''
#     # Add new conflicts to resopairs and confpairs_all and new losses to lospairs_all
#     self.resopairs.update(conf.confpairs)
#     # Conflict pairs to be deleted
#     delpairs = set()
#     changeactive = dict()
#     # smallest relative angle between vectors of heading a and b
#     def anglediff(a, b):
#         d = a - b
#         if d > 180:
#             return anglediff(a, b + 360)
#         elif d < -180:
#             return anglediff(a + 360, b)
#         else:
#             return d
        
#     # Look at all conflicts, also the ones that are solved but CPA is yet to come
#     for conflict in self.resopairs:
#         idx1, idx2 = bs.traf.id2idx(conflict)
#         # If the ownship aircraft is deleted remove its conflict from the list
#         if idx1 < 0:
#             delpairs.add(conflict)
#             continue
#         if idx2 >= 0:
#             # Distance vector using flat earth approximation
#             re = 6371000.
#             dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) 
#                                   np.cos(0.5 * np.radians(intruder.lat[idx2] +
#                                                           ownship.lat[idx1])),
#                                   np.radians(intruder.lat[idx2] - ownship.lat[idx1])]
#             # Relative velocity vector
#             vrel = np.array([intruder.gseast[idx2] - ownship.gseast[idx1],
#                              intruder.gsnorth[idx2] - ownship.gsnorth[idx1]])
#             # Check if conflict is past CPA
#             past_cpa = np.dot(dist, vrel) > 0.0
#             rpz = np.max(conf.rpz[[idx1, idx2]])
#             # hor_los:
#             # Aircraft should continue to resolve until there is no horizontal
#             # LOS. This is particularly relevant when vertical resolutions
#             # are used.
#             hdist = np.linalg.norm(dist)
#             hor_los = hdist < rpz
#             # Bouncing conflicts:
#             # If two aircraft are getting in and out of conflict continously,
#             # then they it is a bouncing conflict. ASAS should stay active until
#             # the bouncing stops.
#             is_bouncing = \
#                 abs(anglediff(ownship.trk[idx1], intruder.trk[idx2])) < 30.0 and \
#                 hdist < rpz * self.resofach
#         # Start recovery for ownship if intruder is deleted, or if past CPA
#         # and not in horizontal LOS or a bouncing conflict
#         if idx2 >= 0 and (not past_cpa or hor_los or is_bouncing):
#             # Enable ASAS for this aircraft
#             changeactive[idx1] = True
#         else:
#             # Switch ASAS off for ownship if there are no other conflicts
#             # that this aircraft is involved in.
#             changeactive[idx1] = changeactive.get(idx1, False)
#             # If conflict is solved, remove it from the resopairs list
#             delpairs.add(conflict)
#     for idx, active in changeactive.items():
#         # Loop a second time: this is to avoid that ASAS resolution is
#         # turned off for an aircraft that is involved simultaneously in
#         # multiple conflicts, where the first, but not all conflicts are
#         # resolved.
#         self.active[idx] = active
#         if not active:
#             # Waypoint recovery after conflict: Find the next active waypoint
#             # and send the aircraft to that waypoint.
#             iwpid = bs.traf.ap.route[idx].findact(idx)
#             if iwpid != -1:  # To avoid problems if there are no waypoints
#                 bs.traf.ap.route[idx].direct(
#                     idx, bs.traf.ap.route[idx].wpname[iwpid])
#     # Remove pairs from the list that are past CPA or have deleted aircraft
#     self.resopairs -= delpairs