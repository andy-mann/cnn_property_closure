a = prop[:,0]
b = prop[:,1]
zero = np.logical_and(a>5176, a<5455)
one = np.logical_and(b>655, b<785)
idx = np.logical_and(zero,one)
p1 = a[idx]
p2= b[idx]
dat = np.stack((p1,p2),1)

----------------------------------------------------------------------

fig,ax = plt.subplots(1,3)
ax[0].imshow(base[2,:,:,15].real)
ax[0].set_title('base point 2')
ax[1].imshow(base[12,:,:,15].real)
ax[1].set_title('base point 12')
ax[2].imshow(re[41,:,:,15])
ax[2].set_title('extrapolate')

----------------------------------------------------------------------

fig,ax=plt.subplots(2,3)

ax[0,0].imshow(train_m[10,0,:,:,15], extent=(0,31,0,31))
ax[0,0].tick_params(left=False,bottom=False, labelleft=False,labelbottom=False)
ax[0,1].imshow(train_m[0,0,:,:,15], extent=(0,31,0,31))
ax[0,1].tick_params(left=False,bottom=False, labelleft=False,labelbottom=False)
ax[0,2].imshow(train_m[12287,0,:,:,15], extent=(0,31,0,31))
ax[0,2].tick_params(left=False,bottom=False, labelleft=False,labelbottom=False)

i1 = ax[1,0].imshow(train_s[10,:,:,15].real, extent=(-15,15,-15,15))
plt.colorbar(i1, ax=ax[1,0])
i2 = ax[1,1].imshow(train_s[0,:,:,15].real, extent=(-15,15,-15,15))
plt.colorbar(i2, ax=ax[1,1])
i3 = ax[1,2].imshow(train_s[12287,:,:,15].real, extent=(-15,15,-15,15))
plt.colorbar(i3, ax=ax[1,2])

----------------------------------------------------------------------

fig,ax=plt.subplots(2,2)

ax[0,0].imshow(mm[1,0,:,:,51], extent=(0,101,0,101))
ax[0,0].tick_params(left=False,bottom=False, labelleft=False,labelbottom=False)
ax[1,0].imshow(mm[10,0,:,:,51], extent=(0,101,0,101))
ax[1,0].tick_params(left=False,bottom=False, labelleft=False,labelbottom=False)


i1 = ax[0,1].imshow(stats101[1,:,:,51].real, extent=(-50,50,-50,50))
plt.colorbar(i1, ax=ax[0,1])
i2 = ax[1,1].imshow(stats101[10,:,:,51].real, extent=(-50,50,-50,50))
plt.colorbar(i2, ax=ax[1,1])

----------------------------------------------------------------------

fig,ax=plt.subplots(2,2)

ax[0,0].imshow(m[1,0,:,:,26])
ax[0,0].tick_params(left=False,bottom=False, labelleft=False,labelbottom=False)
ax[1,0].imshow(m[10,0,:,:,26])
ax[1,0].tick_params(left=False,bottom=False, labelleft=False,labelbottom=False)


i1 = ax[0,1].imshow(stat51[1,:,:,26].real, extent=(-25,25,-25,25))
plt.colorbar(i1, ax=ax[0,1])
i2 = ax[1,1].imshow(stat51[10,:,:,26].real, extent=(-25,25,-25,25))
plt.colorbar(i2, ax=ax[1,1])
plt.tight_layout()

----------------------------------------------------------------------

fig, ax = plt.subplots(3, 3)

ax[0, 0].imshow(origin[0, :, :, 15].real, extent=(-15,15,-15,15))
ax[0,1].imshow(origin[0, :, 15,:].real, extent=(-15,15,-15,15))
c1 = ax[0,2].imshow(origin[0,15,:,:].real, extent=(-15,15,-15,15))
plt.colorbar(c1,ax=ax[0,2])

ax[1,0].imshow(mean_stats[0, :, :, 15].real, extent=(-15,15,-15,15))
ax[1,1].imshow(mean_stats[0, :, 15,:].real, extent=(-15,15,-15,15))
c2 = ax[1,2].imshow(mean_stats[0, 15,:,:].real, extent=(-15,15,-15,15))
plt.colorbar(c2,ax=ax[1,2])

ax[2, 0].imshow(origin[1, :, :, 15].real, extent=(-15,15,-15,15))
ax[2,1].imshow(origin[1, :, 15,:].real, extent=(-15,15,-15,15))
c1 = ax[2,2].imshow(origin[1,15,:,:].real, extent=(-15,15,-15,15))
plt.colorbar(c1,ax=ax[2,2])

plt.show()

----------------------------------------------------------------------

plt.scatter([0, 8067], [0, 2307], c="k")

plt.scatter(origin[:, 0], origin[:, 1], label="Original")
plt.scatter(fea[:, 0], fea[:, 1], label="FEA")
plt.scatter(cnn[:, 0], cnn[:, 1], label="CNN")
plt.scatter(mean_fea[0], mean_fea[1], label="FEA Mean")
plt.scatter(mean_cnn[0], mean_cnn[1], label="CNN Mean")
plt.scatter(avg_stat_pred[:, 0], avg_stat_pred[:, 1], label="Avg Statistics CNN")
plt.xlabel(r"$\displaystyle C_{1111}$")
plt.ylabel(r"$\displaystyle C_{1212}$")
plt.legend()

plt.show()