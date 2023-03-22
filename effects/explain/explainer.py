import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import cycle
from itertools import accumulate, groupby
from scipy.stats import norm
from scipy.stats import ks_2samp
from statistics import NormalDist
from tqdm import tqdm
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler


from effects.extract.aggregations import MaxAgg, MinAgg, MeanAgg, PtpAgg, TrendAgg, StdAgg, SumAgg, MedianAgg, VarAgg, PeakAgg
from effects.extract.conf import *

AGGREGATION_DICT = {'min': MinAgg(),
                    'max': MaxAgg(),
                    'trend': TrendAgg(),
                    'std' : StdAgg(),
                    'mean': MeanAgg(),
                    'var': VarAgg(),
                    'median': MedianAgg(),
                    'sum': SumAgg(),
                    'peak': PeakAgg()}



class Explainer:

    def __init__(self, extractor):
        self.extractor = extractor

        f = pd.DataFrame(extractor.feature_vector.columns, columns=['feature'])
        f['temp'] = f['feature'].apply(lambda x: x.split('__'))
        f['bt'] = f['temp'].apply(lambda x: x[4] if len(x) == 8 else np.nan)
        f['col1'] = f['temp'].apply(lambda x: x[0])
        f['col1_ut'] = f['temp'].apply(lambda x: x[1])
        f['col2'] = f['temp'].apply(lambda x: x[2] if len(x) == 8 else np.nan)
        f['col2_ut'] = f['temp'].apply(lambda x: x[3] if len(x) == 8 else np.nan)
        f['slice_b'] = f['temp'].apply(lambda x: x[-3])
        f['slice_e'] = f['temp'].apply(lambda x: x[-2])
        f['slice_id'] = f['slice_b'].map(str) + '_' + f['slice_e'].map(str)
        f['agg_f'] = f['temp'].apply(lambda x: x[-1])
        f = f.drop('temp', axis=1)

        #         feature_importances_ = pd.DataFrame(np.array([extractor.feature_vector.columns, clf.feature_importances_]).T,
        #                                                  columns=['feature', 'importance'])
        #         self.features = pd.merge(f, feature_importances_, on='feature').sort_values('importance', ascending=False)
        self.features = pd.DataFrame(np.array([extractor.feature_vector.columns]).T, columns=['feature'])
        self.features = pd.merge(self.features, f, on='feature')

        self.color_dict = {cls: color for cls, color in
                           zip(np.unique(self.extractor.y_train), cycle(sns.color_palette()))}

        pairs = [(a, b) for idx, a in enumerate(np.unique(self.extractor.y_train))
                 for b in np.unique(self.extractor.y_train)[idx + 1:]]
        distances = {}
        for cls1, cls2 in pairs:
            distances['__'.join([cls1, cls2])] = []
        for col in tqdm(self.features['feature']):
            #             distibutions = {}
            #             for cls in np.unique(self.extractor.y_train):
            #                 mu, std = norm.fit(df_train[y_train == cls][col].values)
            #                 distibutions[cls] = (mu + 1e-9, std + 1e-9)
            #             for cls1, cls2 in pairs:
            #                 overlap = NormalDist(mu=distibutions[cls1][0], sigma=distibutions[cls1][1]). \
            #                                     overlap(NormalDist(mu=distibutions[cls2][0], sigma=distibutions[cls2][1]))
            #                 distances['__'.join([cls1, cls2])].append(1-overlap)
            for cls1, cls2 in pairs:
                a = self.extractor.feature_vector[self.extractor.y_train == cls1][col].values
                b = self.extractor.feature_vector[self.extractor.y_train == cls2][col].values
                scaler = MinMaxScaler().fit(np.concatenate([a, b]).reshape(-1, 1))
                distances['__'.join([cls1, cls2])].append(wasserstein_distance
                                                          (scaler.transform(a.reshape(-1, 1)).reshape(-1),
                                                           scaler.transform(b.reshape(-1, 1)).reshape(-1)))

        for cls1, cls2 in pairs:
            #             self.features['__'.join([cls1, cls2])] = MinMaxScaler().fit_transform(
            #                 np.array(distances['__'.join([cls1, cls2])]).reshape(-1, 1)).reshape(-1)
            self.features['__'.join([cls1, cls2])] = np.array(distances['__'.join([cls1, cls2])])
        self.features['score'] = self.features[self.features.columns[-len(pairs):]].sum(axis=1)

        for y in np.unique(self.extractor.y_train):
            columns = [c for c in self.features.columns if '__' in c and y in c]
            self.features['class_score_' + y] = self.features[columns].sum(axis=1) / len(columns)

        self.features = self.features.sort_values('score', ascending=False)

    def title(self, f):
        return 'Dimension: {0}, Transformation: {1}, Slice: {2}, Agg: {3} \n Score: {4}'.format(f['col1'], f['col1_ut'], \
                                                                                                f['slice_id'].replace(
                                                                                                    '_', '-'),
                                                                                                f['agg_f'],
                                                                                                str(round(
                                                                                                    float(f['score']),
                                                                                                    2)))

    def __plot_bar_plots(self, col, **kwargs):
        plt.figure(figsize=(10, 10))
        ut_importance = self.features.groupby(col, dropna=False)['score'].mean().reset_index().fillna('None')

        plots = sns.barplot(x=col, y='score', data=ut_importance)

        for bar in plots.patches:
            plots.annotate(format(bar.get_height(), '.2f'),
                           (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                           size=10, xytext=(0, 5),
                           textcoords='offset points')

        title_dict = {'col1_ut': 'Univariate Transform', 'bt': 'Bivaraite Transform', 'agg_f': 'Aggregation Function'}
        plt.title(title_dict[col] + ' Score', fontsize=TITLE_FONTSIZE)
        plt.xlabel(title_dict[col], fontsize=LABEL_FONTSIZE)
        plt.ylabel('Score', fontsize=LABEL_FONTSIZE)
        plt.plot()

    def plot_univariate_transform_importance(self, **kwargs):
        self.__plot_bar_plots('col1_ut', **kwargs)

    def plot_bivariate_transform_importance(self, **kwargs):
        self.__plot_bar_plots('bt', **kwargs)

    def plot_aggregation_importance(self, **kwargs):
        self.__plot_bar_plots('agg_f', **kwargs)

    def plot_time_scores(self):
        ts = self.extractor.slicer.time_scores
        time_scores = ts if ts != [] else np.ones(self.extractor.df.values[0][0].size)
        plt.plot(time_scores)

    def plot_trainset(self, n_samples=3, **kwargs):
        columns = [col for col in self.extractor.df.columns if col.count(SEPERATOR + 'identity') == 1]
        fig, axs = plt.subplots(len(columns), figsize=(10, 8))
        fig.suptitle('Train set samples', fontsize=TITLE_FONTSIZE)
        if len(columns) == 1:
            axs = [axs]
        for i in range(len(columns)):
            for cls in np.unique(self.extractor.y_train):
                flag = True
                for ts in self.extractor.df[columns[i]][self.extractor.y_train == cls].sample(n_samples,
                                                                                              replace=True).values:
                    axs[i].plot(ts, color=self.color_dict[cls], label=cls if (i == 0 and flag) else "")
                    flag = False
            axs[i].grid()
            axs[i].set_ylabel(columns[i].split(SEPERATOR)[0], rotation=0, labelpad=len(columns[i]) * 3,
                              fontsize=LABEL_FONTSIZE)
        plt.xlabel('Time', fontsize=LABEL_FONTSIZE)
        plt.subplots_adjust(hspace=0)
        if len(columns) == 1:
            axs[0].legend()
        else:
            fig.legend()
        plt.show()

    def plot_feature(self, feature, classes=[], n_samples=3, alpha_raw=0.2, alpha_trns=1,
                     zoom_lines=True, **kwargs):
        if classes == []:
            classes = np.unique(self.extractor.y_train)

        columns = [col for col in self.extractor.df.columns if col.count(SEPERATOR + 'identity') == 1]
        feature_details = self.features[self.features['feature'] == feature].iloc[0]
        dim = feature_details['col1'] + SEPERATOR + 'identity'
        dim_and_transform = feature_details['col1'] + SEPERATOR + feature_details['col1_ut']
        sb = int(feature_details['slice_b'])
        se = int(feature_details['slice_e'])
        length = len(self.extractor.df.values[0][0])
        max_x = np.stack(self.extractor.df[dim].values).max()
        min_x = np.stack(self.extractor.df[dim].values).min()
        bt = feature_details['bt']
        is_bt = pd.notna(bt)
        if is_bt:
            dim_and_transform2 = feature_details['col2'] + SEPERATOR + feature_details['col2_ut']
            transformed_dim_name = dim_and_transform + SEPERATOR + dim_and_transform2 + SEPERATOR + bt

        fig = plt.figure(figsize=(12, 10))
        plt.subplots_adjust(bottom=0., left=0, top=1, right=1)

        fig.suptitle(self.title(feature_details), fontsize=TITLE_FONTSIZE, y=1.1)

        sub1 = fig.add_subplot(2, 10, (11, 18))  # raw data
        sub2 = fig.add_subplot(2, 10, 19)  # slice importance
        sub3 = fig.add_subplot(2, 3, 1)  # zoom in
        sub4 = fig.add_subplot(2, 3, 2)
        sub5 = fig.add_subplot(2, 3, 3)

        slices_score = self.features.groupby('slice_id')['score'].sum().to_dict()
        score_map = np.zeros(length)
        for k, v in slices_score.items():
            if k != feature_details['slice_id']:
                v = 0
            score_map[int(k.split('_')[0]):int(k.split('_')[1])] = v
        score_map = np.expand_dims(score_map, axis=0)

        data_to_plot = {}
        for cls in classes:
            data_to_plot[cls] = (self.extractor.df[self.extractor.y_train == cls].sample(n_samples)).index

        if is_bt:
            dim2 = feature_details['col2'] + SEPERATOR + 'identity'
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=self.color_dict[cls], lw=2))
                for ts in self.extractor.df[dim].loc[data_to_plot[cls]].values:
                    sub1.plot(ts, color=self.color_dict[cls])
                for ts in self.extractor.df[dim2].loc[data_to_plot[cls]].values:
                    sub1.plot(ts, linestyle='dotted', color=self.color_dict[cls])
            leg = sub1.legend(custom_lines, classes, loc='upper right', fontsize=9)
            sub1.add_artist(leg)
            legend2 = [Line2D([0], [0], linestyle='solid', color='black', label=feature_details['col1'],
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label=feature_details['col2'], linestyle='dotted',
                              markersize=9)]
            sub1.legend(handles=legend2, loc='upper left')
            sub1.set_title('Raw Data', fontsize=TITLE_FONTSIZE)
            sub1.set_xlabel('Time', fontsize=LABEL_FONTSIZE)
            sub1.grid()

        else:
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=self.color_dict[cls], lw=2))
                for ts in self.extractor.df[dim].loc[data_to_plot[cls]].values:
                    sub1.plot(ts, color=self.color_dict[cls])
            sub1.legend(custom_lines, classes, loc='upper right')
            sub1.set_title('Raw Data', fontsize=TITLE_FONTSIZE)
            sub1.set_xlabel('Time', fontsize=LABEL_FONTSIZE)
            sub1.set_ylabel(feature_details['col1'], fontsize=LABEL_FONTSIZE)
            sub1.grid()

        cb = sub1.imshow(score_map, extent=[0.0, length, min_x * 1.1, max_x * 1.1], cmap='binary',
                         interpolation='nearest', aspect='auto',
                         alpha=0.9)

        # cb_ax = fig.add_axes([.92,.0,.05,.45])
        clb = fig.colorbar(cb, cax=sub2)
        clb.ax.set_ylabel('Slice Importance', rotation=270, labelpad=20, fontsize=20)

        sub3.clear()
        if is_bt:
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=self.color_dict[cls], lw=2))
                for ts in self.extractor.df.loc[data_to_plot[cls]][dim_and_transform].values:
                    sub3.plot(np.arange(sb, se), ts[sb:se], color=self.color_dict[cls], alpha=alpha_raw)
                for ts in self.extractor.df.loc[data_to_plot[cls]][dim_and_transform2].values:
                    sub3.plot(np.arange(sb, se), ts[sb:se], linestyle='dotted', color=self.color_dict[cls],
                              alpha=alpha_raw)
                for ts in self.extractor.df.loc[data_to_plot[cls]][transformed_dim_name].values:
                    sub3.plot(np.arange(sb, se), ts[sb:se], '--', color=self.color_dict[cls], alpha=alpha_trns)

            leg = sub3.legend(custom_lines, classes, loc='upper left', ncol=1, fontsize=int(LABEL_FONTSIZE / 1.5))
            sub3.add_artist(leg)
            sub3.set_title('Zoom into slice [' + str(sb) + ', ' + str(se) + ')', fontsize=int(TITLE_FONTSIZE * 0.75))
            sub3.set_xlabel('Time', fontsize=int(LABEL_FONTSIZE * 0.75))
            sub3.grid()
            legend2 = [Line2D([0], [0], color='black', label=dim_and_transform,
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label=dim_and_transform2, linestyle='dotted',
                              markersize=9),
                       Line2D([0], [0], color='black', lw=2, label=bt, linestyle='dashed', markersize=9)]
            sub3.legend(handles=legend2, loc='upper right')


        else:
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=self.color_dict[cls], lw=2))
                for ts in self.extractor.df.loc[data_to_plot[cls]][dim].values:
                    sub3.plot(np.arange(sb, se), ts[sb:se], '--', color=self.color_dict[cls], alpha=alpha_raw)
                for ts in self.extractor.df.loc[data_to_plot[cls]][dim_and_transform].values:
                    sub3.plot(np.arange(sb, se), ts[sb:se], color=self.color_dict[cls], alpha=alpha_trns)

            leg = sub3.legend(custom_lines, classes, loc='upper left', ncol=1, fontsize=int(LABEL_FONTSIZE / 1.5))
            sub3.add_artist(leg)
            sub3.set_title('Zoom into slice [' + str(sb) + ', ' + str(se) + ')', fontsize=int(TITLE_FONTSIZE * 0.75))
            sub3.set_xlabel('Time', fontsize=int(LABEL_FONTSIZE * 0.75))
            sub3.grid()
            legend2 = [Line2D([0], [0], color='black', label='Before ' + feature_details['col1_ut'], linestyle='dashed',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='After ' + feature_details['col1_ut'], markersize=9)]
            sub3.legend(handles=legend2, loc='upper right')

        data = {}
        data_dim = dim_and_transform if not is_bt else transformed_dim_name
        for cls in classes:
            data[cls] = []
            for ts in self.extractor.df.loc[data_to_plot[cls]][data_dim].values:
                data[cls].append(ts)
        AGGREGATION_DICT[feature_details['agg_f']].explain()(classes, sub4, sb, se, data,
                                                                     self.color_dict,
                                                                     feature_details['agg_f'] + ' visualization', '',
                                                                     int(TITLE_FONTSIZE * 0.75),
                                                                     int(LABEL_FONTSIZE * 0.75))

        for cls in classes:
            sns.distplot(self.extractor.feature_vector[feature][self.extractor.y_train == cls],
                         color=self.color_dict[cls], \
                         ax=sub5, label=cls)
        sub5.legend(fontsize=8)
        sub5.set_title('Feature Distribution', fontsize=int(TITLE_FONTSIZE * 0.75))

        if zoom_lines:
            min_x_axis, max_x_axis = sub1.transData.inverted().transform(sub1.bbox)[:, 1]
            con1 = ConnectionPatch(xyA=(sb, min_x_axis), coordsA=sub1.transData,
                                   xyB=(sb, max_x_axis), coordsB=sub1.transData, color='black', linewidth=3)
            fig.add_artist(con1)

            con2 = ConnectionPatch(xyA=(sb, max_x_axis), coordsA=sub1.transData,
                                   xyB=(se, max_x_axis), coordsB=sub1.transData, color='black', linewidth=3)
            fig.add_artist(con2)

            con3 = ConnectionPatch(xyA=(se, max_x_axis), coordsA=sub1.transData,
                                   xyB=(se, min_x_axis), coordsB=sub1.transData, color='black', linewidth=3)
            fig.add_artist(con3)

            con4 = ConnectionPatch(xyA=(sb, min_x_axis), coordsA=sub1.transData,
                                   xyB=(se, min_x_axis), coordsB=sub1.transData, color='black', linewidth=3)
            fig.add_artist(con4)

    def get_seperatetors(self, class_1, class_2):
        class_1, class_2 = sorted([class_1, class_2])
        return self.features.sort_values('__'.join([class_1, class_2]), ascending=False)['feature'].tolist()

    def plot_best_seperator(self, class_1, class_2, k=0, **kwargs):
        class_1, class_2 = sorted([class_1, class_2])
        kwargs['classes'] = [class_1, class_2]
        self.plot_feature(self.get_seperatetors(class_1, class_2)[k], **kwargs)

    def plot_best_feature(self, k=0, n_samples=3, **kwargs):
        self.features = self.features.sort_values('score', ascending=False)
        self.plot_feature(self.features['feature'].iloc[k], n_samples=n_samples, **kwargs)

    def plot_slices_scores(self, **kwargs):
        length = len(self.extractor.df.values[0][0])
        plt.figure(figsize=(12, 8))
        slices_score = self.features.groupby('slice_id')['score'].sum().to_dict()
        score_map = np.zeros(length)
        for k, v in slices_score.items():
            score_map[int(k.split('_')[0]):int(k.split('_')[1])] = v
        score_map = np.expand_dims(score_map, axis=0)
        plt.imshow(score_map, cmap='binary', interpolation='nearest', aspect='auto')
        plt.yticks([])
        plt.ylabel('Score', fontsize=LABEL_FONTSIZE)
        plt.xlabel('Time', fontsize=LABEL_FONTSIZE)
        plt.title('Slices Scores', fontsize=TITLE_FONTSIZE)
        plt.colorbar()
        plt.show()

    def plot_slice_seperation(self, slice_id, **kwargs):
        plt.figure(figsize=(10, 11))
        labels = np.unique(self.extractor.y_train)
        n = len(labels)
        imap = np.zeros((n, n))
        for c1 in range(n - 1):
            for c2 in range(c1 + 1, n):
                s = self.features[ex.features['slice_id'] == slice_id]['__'.join([labels[c1], labels[c2]])].max()
                imap[c1][c2] = s
                imap[c2][c1] = s

        plt.figure(figsize=(7, 7))
        sns.heatmap(imap, cmap='PuBuGn', xticklabels=labels, yticklabels=labels, vmax=1)
        plt.title('Slice [{0}, {1}) seperateion score'.format(slice_id.split('_')[0], slice_id.split('_')[1]))
        plt.show()

    def plot_class_scores(self, k=3, classes=[], **kwargs):
        length = len(self.extractor.df.values[0][0])
        if classes == []:
            classes = np.unique(self.extractor.y_train)
        fig = plt.figure(figsize=(10, 8))
        plt.title('Class Scores - top ' + str(k) + ' features', fontsize=TITLE_FONTSIZE)
        plt.ylabel('Score', fontsize=LABEL_FONTSIZE)
        plt.xlabel('Time', fontsize=LABEL_FONTSIZE)
        for cls in classes:
            for _, ts in self.features.sort_values('class_score_' + cls, ascending=False).iloc[:k].iterrows():
                sb = int(ts['slice_b'])
                se = int(ts['slice_e'])
                score = float(ts['class_score_' + cls])
                plt.plot([sb, se], [score, score], color=self.color_dict[cls], alpha=0.9, linewidth=4)
        plt.grid()
        custom_lines = []
        for cls in np.unique(self.extractor.y_train):
            custom_lines.append(Line2D([0], [0], color=self.color_dict[cls], lw=2))
        leg = plt.legend(custom_lines, np.unique(self.extractor.y_train), loc='best', fontsize=9)
        fig.add_artist(leg)
        plt.ylim(0.2, 0.9)  ##########################
        for cls in classes:
            for ts in self.extractor.df[self.extractor.df.columns[0]][self.extractor.y_train == cls].values[:3]:
                ts = MinMaxScaler((0.21, 0.39)).fit_transform(ts.reshape(-1, 1)).reshape(-1)
                plt.plot(ts, color=self.color_dict[cls], alpha=0.6)
        plt.axvspan(0, length, 0, 0.3, color='black', alpha=0.1)

    def plot_feature_seperations(self, feature, **kwargs):
        plt.figure(figsize=(10, 11))
        scores = self.features[self.features['feature'] == feature].iloc[0]
        labels = np.unique(self.extractor.y_train)
        n = len(labels)
        imap = np.zeros((n, n))
        for c1 in range(n - 1):
            for c2 in range(c1 + 1, n):
                imap[c1][c2] = scores['__'.join([labels[c1], labels[c2]])]
                imap[c2][c1] = scores['__'.join([labels[c1], labels[c2]])]

        plt.figure(figsize=(7, 7))
        sns.heatmap(imap, cmap='PuBuGn', xticklabels=labels, yticklabels=labels, vmax=1)
        plt.title('{0} seperateion score'.format(feature))

    def extract_features(self, k=3):
        features = []
        for col in [c for c in self.features.columns if '__' in c]:
            features += self.features[['feature', col]].sort_values(col, ascending=False).iloc[:k]['feature'].tolist()
        return list(set(features))