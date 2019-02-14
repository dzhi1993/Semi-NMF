import scipy.io
data1 = scipy.io.loadmat("flatmap_vertices.mat")
data2 = scipy.io.loadmat("flatmap_edges.mat")

#
# for i in data1:
#     if '__' not in i and 'readme' not in i:
#           np.savetxt(("flatmap_vertices.csv"), data1[i], delimiter=',')
#
# for j in data2:
#     if '__' not in j and 'readme' not in j:
#           np.savetxt(("flatmap_edges.csv"), data2[j], delimiter=',')

csv_data1 = pd.read_csv("flatmap_vertices.csv")
csv_data1 = csv_data1.drop(csv_data1.columns[[2]], axis=1)
csv_data1 = csv_data1 / 100
print(csv_data1.info(), csv_data1.describe(), csv_data1)
np.savetxt(("test_vertices.csv"), csv_data1, delimiter=',', newline=',')

csv_data2 = pd.read_csv("flatmap_edges.csv")
print(csv_data2.info(), csv_data2.describe(), csv_data2)
