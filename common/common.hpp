#define h(i, j) h[(i) * (ny + 1) + (j)]
#define h_all(i, j) h_all[(i) * (ny + 1) + (j)]
#define u(i, j) u[(i) * (ny) + (j)]
#define u_all(i, j) u_all[(i) * (ny) + (j)]
#define v(i, j) v[(i) * (ny + 1) + (j)]

#define dh(i, j) dh[(i) * ny + (j)]
#define du(i, j) du[(i) * ny + (j)]
#define dv(i, j) dv[(i) * ny + (j)]

#define dh1(i, j) dh1[(i) * ny + (j)]
#define du1(i, j) du1[(i) * ny + (j)]
#define dv1(i, j) dv1[(i) * ny + (j)]

#define dh2(i, j) dh2[(i) * ny + (j)]
#define du2(i, j) du2[(i) * ny + (j)]
#define dv2(i, j) dv2[(i) * ny + (j)]

#define dh_dx(i, j) (h(i + 1, j) - h(i, j)) / dx
#define dh_dy(i, j) (h(i, j + 1) - h(i, j)) / dy

#define du_dx(i, j) (u(i + 1, j) - u(i, j)) / dx
#define dv_dy(i, j) (v(i, j + 1) - v(i, j)) / dy