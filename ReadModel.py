import h5py
from sklearn import decomposition


class ReadModel:

    def __init__(self, layer_name_common, modeldir, apply_pca, n_components=500):
        self.layer_name_common = layer_name_common
        self.modeldir = modeldir
        self.folder_name = 'HvM_forward_features/'
        self.feature_type = 'forward-outlayer'
        self.apply_pca = apply_pca
        self.n_components = n_components

    def get_model(self):
        layer_names_common = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
        netF_names = ['stage0', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7']

        if self.layer_name_common[0:6] == 'resnet':
            layer_name = self.layer_name_common
        else:
            stage_id = layer_names_common.index(self.layer_name_common)
            layer_name = netF_names[stage_id]

        fname_model_features = 'model_features_%s_%s.h5' % (self.feature_type, layer_name)
        h5f = h5py.File(self.modeldir + self.folder_name + fname_model_features, 'r')

        model_features = h5f[fname_model_features[:-3]][:]
        h5f.close()

        if self.apply_pca:
            pca = decomposition.PCA(self.n_components)

            pca = decomposition.PCA(self.n_components)
            pca.fit(model_features) #[train_inds, :]
            model_features = pca.transform(model_features)
        return model_features
