import numpy as np 

points=[[2,2],[1,2.75],[2,3.5],[1,4.25],[2,5.5],[1,6],[4,2],[3,2.75],[4,4.25],[3,4.75],[4,5.25],[3,6.25],[6,2],[5,2.75],[6,3.5],[5,4.25],[6,5.5]]

def compute_error_metric(a,b,c,d,er1,er2,er3):
	point1=points[a-1]
	point2=points[b-1]
	point3=points[c-1]
	real_point=points[d-1]

	w1=1/er1
	w2=1/er2
	w3=1/er3

	candid_point_x=(w1*point1[0]+w2*point2[0]+w2*point3[0])/(w1+w2+w3)
	candid_point_y=(w1*point1[1]+w2*point2[1]+w3*point3[1])/(w1+w2+w3)
	candid_point=[candid_point_x,candid_point_y]
	print('candidate point specifics are:')
	print(candid_point)
	real_point_x=real_point[0]
	real_point_y=real_point[1]


	error=np.sqrt(np.square(candid_point_x-real_point_x)+np.square(candid_point_y-real_point_y))
	return error

