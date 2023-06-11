import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, ctx
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.figure_factory as ff
import webbrowser
from threading import Timer

from effects.config import SEPARATOR

def ipeak(x):
    return max(x.min(), x.max(), key=abs)

def peak(X, axis=1):
    if len(X[0]) == 1:
        return np.zeros(len(X))
    return np.array([ipeak(x) for x in X])

def argpeak(x):
    p = ipeak(x)
    return np.where(x == p)[0][0]


SCATTER_FUNCTIONS = {'max': [np.max, np.argmax], 'min': [np.max, np.argmax], 'peak': [ipeak, argpeak]}
LINE_FUNCTIONS = {'mean': np.mean, 'var': np.var, 'std': np.std, 'median': np.median, 'ptp': np.ptp}

class ExplorerApp:
    def __init__(self, explainer):
        self.explainer = explainer
        ex = self.explainer

        df = ex.feature_scores
        df = df.fillna('None')
        df_table = df.drop(['slice_id'], axis=1)
        df_table = np.around(df_table, 3)
        df_table['id'] = df_table['name']
        df_table.set_index('id', inplace=True, drop=False)

        self.app = Dash(external_stylesheets=[dbc.themes.SLATE])

        stats = html.Div([dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    ['Dimension', 'Univariate Transforms', 'Bivariate Transforms', 'Slices', 'Aggregation Function'],
                    'Dimension',
                    id='stats-column'
                )
            ], width=4),
            dbc.Col([
                dcc.RadioItems(
                    ['Sum', 'Mean'],
                    'Sum',
                    id='stats-fig-type',
                    inline=True,
                    inputStyle={"margin-left": "20px"}
                )
            ], width=6),
        ], align='center'),
            dbc.Row([dcc.Graph(id='feature-stats')])
        ])

        feature_scores = html.Div([
            dcc.Graph(id='feature-scores'),
            dcc.Slider(1, 20, 1, id='feature-scores-k', value=3)
        ])

        org_columns_names = sorted(list(set([c.split(SEPARATOR)[0] for c in ex.extractor.transformed_data.columns])))

        trainset = html.Div([dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    org_columns_names,
                    org_columns_names[0],
                    id='trainset-column'
                )
            ], width=4),
            dbc.Col([
                daq.ToggleSwitch(id='show-slices-score',
                                 label='Show Slices Score')
            ], width=6),
        ], align='center'),
            dbc.Row([dcc.Graph(id='trainset-graph'),
                     dcc.Slider(1, 20, 1, value=3, id='trainset-num-samples')])
        ])

        feature_separation = html.Div([
            html.H4('Feature Separation', style={'textAlign': 'center'}),
            dcc.Graph(id='feature-separation')
        ])

        slice_separation = html.Div([dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    ['_'.join([str(s[0]), str(s[1])]) for s in ex.extractor.slices],
                    ['_'.join([str(s[0]), str(s[1])]) for s in ex.extractor.slices][0],
                    id='slices')
            ], width=4),
            dbc.Col([
                dcc.RadioItems(
                    ['max', 'mean', 'sum', 'min'],
                    'max',
                    id='slice-separation-agg',
                    inline=True,
                    inputStyle={"margin-left": "20px"})
            ], width=6),
        ], align='center'),
            dbc.Row([dcc.Graph(id='slice-separation')])
        ])

        score_map = np.zeros(ex.length)
        for k, v in ex.slices_score.items():
            score_map[int(k.split('_')[0]):int(k.split('_')[1])] += v
        slice_scores_fig = px.imshow(np.repeat(score_map.reshape(1, -1), ex.length / 2, axis=0),
                                     color_continuous_scale='Greys', range_color=[0, ex.slice_max_score])
        slice_scores_fig.update_yaxes(title='Score')
        slice_scores_fig.update_xaxes(title='Timestamp')
        slice_scores = html.Div([dbc.Row([
            dbc.Col([html.H4('Slices Scores', style={'textAlign': 'center'})]),
        ]),
            dbc.Row([dcc.Graph(id='slice-score-fig', figure=slice_scores_fig)])
        ])

        feature_visualizer = html.Div([dbc.Row([
            dbc.Col([dcc.Graph(id='feature-visualizer-1', style={'height': '54vh'})], width=4),
            dbc.Col([dcc.Graph(id='feature-visualizer-2', style={'height': '54vh'})], width=4),
            dbc.Col([dcc.Graph(id='feature-visualizer-3', style={'height': '54vh'})], width=4)
        ], align='center'),

            dbc.Row([dbc.Col([daq.ToggleSwitch(id='feature-visualizer-column')], width=4),
                     dbc.Col([dcc.Slider(0, 1, .01, value=1, id='feature-visualizer-alpha', marks=None)], width=4),
                     dbc.Col([daq.ToggleSwitch(id='feature-visualizer-rug')], width=4)
                     ], align='center'),

            dbc.Row([dcc.Slider(0, 20, 1, value=3, id='feature-visualizer-n-samples')], align='center'
                    )])

        self.app.layout = html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([dcc.RadioItems(
                            ['Stats', 'Scores', 'Feature Visualizer', 'Slices Scores', 'Slice Separation'],
                            'Stats',
                            id='fig1-to-show',
                            inline=True,
                            inputStyle={"margin-left": "20px"})], width=5),
                        dbc.Col([html.H1('EFFECTS', style={'textAlign': 'left'})], width=7)
                    ], align='center'),

                    html.Br(),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([], id='fig1'),
                                    dbc.Button("Full Screen", id="open-modal", n_clicks=0),
                                    dbc.Modal(
                                        [
                                            dbc.ModalBody(html.Div([feature_visualizer], id='modal-fv')),
                                            dbc.ModalFooter(
                                                dbc.Button(
                                                    "Close", id="close-modal", className="ms-auto", n_clicks=0
                                                )
                                            ),
                                        ],
                                        id="modal",
                                        fullscreen=True,
                                        is_open=False,
                                    )
                                ]))
                        ], width=7),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5('EFFECTS Features', style={'textAlign': 'center'}),
                                    dash_table.DataTable(
                                        columns=[{'name': i, 'id': i} for i in df_table.columns[:-1]],
                                        data=df_table.to_dict('records'),
                                        page_size=12,
                                        sort_action='native',
                                        style_table={'overflowX': 'auto'},
                                        style_cell={'text-align': 'center'},
                                        id='feature-table'
                                    )
                                ]))
                        ], width=5),
                    ], align='center'),

                    html.Br(),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    trainset
                                ]))
                        ], width=7),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    feature_separation
                                ]))
                        ], width=5),
                    ], align='center'),
                ]), color='dark'
            )
        ])

        @self.app.callback(
            Output('feature-stats', 'figure'),
            Input('stats-column', 'value'),
            Input('stats-fig-type', 'value')
        )
        def update_feature_stats(filter_, fig_type):
            filter_to_col = {'Dimension': 'dim_1', 'Univariate Transforms': 'dim_1_uni_transform',
                             'Bivariate Transforms': 'bi_transform',
                             'Slices': 'slice_id', 'Aggregation Function': 'agg'}

            if filter_ not in list(filter_to_col.keys()):
                filter_ = 'Dimension'

            col = filter_to_col[filter_]
            if fig_type == 'Sum':
                graph_df = pd.DataFrame(ex.feature_scores.groupby(col)['score'].apply(np.sum)).reset_index()
                if filter_ == 'Dimension':
                    graph_df2 = pd.DataFrame(
                        ex.feature_scores.groupby('dim_2')['score'].apply(np.sum)).reset_index().rename(
                        columns={'dim_2': 'dim_1'})
                    graph_df = pd.concat([graph_df, graph_df2]).groupby(['dim_1']).sum().reset_index()
                fig = px.bar(graph_df, x=col, y='score')

            elif fig_type == 'Mean':
                graph_df = pd.DataFrame(ex.feature_scores.groupby(col)['score'].agg([np.mean, np.std])).reset_index()
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=graph_df[col], y=graph_df['mean'],
                    error_y=dict(type='data', array=graph_df['std'])
                ))

            fig.update_layout(title={
                'text': "Feature Scores Stats",
                'y': .95,
                'x': .5,
                'xanchor': 'center',
                'yanchor': 'top'})
            fig.update_xaxes(title=filter_)

            return fig

        @self.app.callback(
            Output('feature-scores', 'figure'),
            Input('feature-scores-k', 'value'),
        )
        def update_feature_scores(k):
            df_to_plot = pd.DataFrame()
            for cls in ex.classes_:
                col = cls + SEPARATOR + 'score'
                df_class = ex.feature_scores.sort_values(col, ascending=False).iloc[:k]
                features_to_plot = pd.DataFrame()
                features_to_plot['Timestamp'] = np.stack(df_class[['slice_start', 'slice_end']].values).reshape(-1)
                features_to_plot['Score'] = np.repeat(df_class[col].values, 2)
                features_to_plot['Feature'] = np.repeat(df_class['name'].values, 2)
                features_to_plot['class'] = cls
                features_to_plot['Dimension'] = np.repeat(df_class['dim_1'].values, 2)
                df_to_plot = pd.concat([df_to_plot, features_to_plot])
            df_to_plot['Timestamp'] = df_to_plot['Timestamp'].astype(int)

            fig = px.line(df_to_plot, x='Timestamp', y='Score', color='class', line_group='Feature',
                          line_dash='Dimension')

            fig.update_xaxes(range=[0, ex.length])

            fig.update_layout(title={
                'text': "Feature Scores - Top " + str(k) + ' per Class',
                'y': .95,
                'x': .5,
                'xanchor': 'center',
                'yanchor': 'top'})

            return fig

        @self.app.callback(
            Output('fig1', 'children'),
            Input('fig1-to-show', 'value')
        )
        def update_fig1(fig1):
            if fig1 == 'Stats':
                return stats
            elif fig1 == 'Slice Separation':
                return slice_separation
            elif fig1 == 'Slices Scores':
                return slice_scores
            elif fig1 == 'Feature Visualizer':
                return feature_visualizer
            else:
                return feature_scores

        @self.app.callback(
            Output('trainset-graph', 'figure'),
            Input('trainset-column', 'value'),
            Input('trainset-num-samples', 'value'),
            Input('show-slices-score', 'value')
        )
        def update_trainset_graph(column, n_samples, show_slices_score):
            index_to_plot = []
            num_lines = n_samples * len(ex.classes_)
            for cls in ex.classes_:
                index_to_plot += list(
                    ex.extractor.transformed_data[ex.extractor.y == cls].sample(n_samples, replace=False).index)
            line_df = pd.DataFrame(columns=['timestamp', 'value', 'class'])
            line_df['value'] = np.stack(
                ex.extractor.transformed_data[column + SEPARATOR + ex.identity].loc[index_to_plot].values).reshape(-1)
            line_df['class'] = np.repeat(ex.extractor.y[index_to_plot], ex.length)
            line_df['timestamp'] = np.repeat(np.arange(ex.length).reshape(1, ex.length), num_lines, axis=0).reshape(-1)
            line_df['id'] = np.repeat(np.arange(num_lines), ex.length)
            fig = px.line(line_df, x='timestamp', y='value', color='class', line_group='id')

            fig.update_layout(title={
                'text': "Train Set",
                'y': .95,
                'x': .5,
                'xanchor': 'center',
                'yanchor': 'top'})
            fig.update_yaxes(title=column)

            if show_slices_score:
                for sl, score in ex.slices_score.items():
                    a, b = sl.split('_')
                    a = int(a)
                    b = int(b)
                    fig.add_vrect(a, b, fillcolor='Grey', opacity=score / ex.slice_max_score)

            return fig

        @self.app.callback(
            Output("feature-separation", "figure"),
            Input("feature-table", "active_cell")
        )
        def feature_separation(active_cell):
            if active_cell is None:
                name = ex.feature_scores['name'].iloc[0]
            else:
                name = active_cell['row_id']
            feature = ex.feature_scores[ex.feature_scores['name'] == name].iloc[0]
            scores_map = np.zeros((ex.num_classes, ex.num_classes))
            for c1 in range(ex.num_classes - 1):
                for c2 in range(c1 + 1, ex.num_classes):
                    scores_map[c1][c2] = feature[SEPARATOR.join([ex.classes_[c1], ex.classes_[c2]])]
                    scores_map[c2][c1] = feature[SEPARATOR.join([ex.classes_[c1], ex.classes_[c2]])]
            scores_map = np.around(scores_map, 3)

            fig = px.imshow(scores_map, text_auto=True, x=ex.classes_, y=ex.classes_, color_continuous_scale='RdBu_r',
                            range_color=[0, 1])
            return fig

        @self.app.callback(
            Output('slice-separation', 'figure'),
            Input('slices', 'value'),
            Input('slice-separation-agg', 'value')
        )
        def update_slice_separation(slice_id, agg):
            m = 1 if agg != 'sum' else ex.slice_max_score / ex.feature_max_score
            agg = {'max': np.max, 'mean': np.mean, 'sum': np.sum, 'min': np.min}[agg]
            scores_map = np.zeros((ex.num_classes, ex.num_classes))
            imap = np.zeros((ex.num_classes, ex.num_classes))
            for c1 in range(ex.num_classes - 1):
                for c2 in range(c1 + 1, ex.num_classes):
                    s = agg(ex.feature_scores[ex.feature_scores['slice_id'] == slice_id][
                                SEPARATOR.join([ex.classes_[c1], ex.classes_[c2]])])
                    scores_map[c1][c2] = s
                    scores_map[c2][c1] = s
            scores_map = np.around(scores_map, 3)
            fig = px.imshow(scores_map, text_auto=True, x=ex.classes_, y=ex.classes_, color_continuous_scale='RdBu_r',
                            range_color=[0, m])
            return fig

        @self.app.callback(
            Output("feature-visualizer-1", "figure"),
            Output("feature-visualizer-2", "figure"),
            Output("feature-visualizer-3", "figure"),
            Input("feature-table", "active_cell"),
            Input("feature-visualizer-n-samples", "value"),
            Input("feature-visualizer-alpha", "value"),
            Input("feature-visualizer-column", "value"),
            Input("feature-visualizer-rug", "value")
        )
        def update_feature_visualizer(active_cell, n_samples, alpha, column, rug):
            if active_cell is None:
                feature = ex.feature_scores.iloc[0]
            else:
                feature = ex.feature_scores[ex.feature_scores['name'] == active_cell['row_id']].iloc[0]

            feature = feature.replace('None', ex.identity)
            slice_start = int(feature['slice_start'])
            slice_end = int(feature['slice_end'])
            length = slice_end - slice_start
            num_lines = n_samples * len(ex.classes_)
            agg = feature['agg']

            index_to_plot = []
            for cls in ex.classes_:
                index_to_plot += list(ex.extractor.transformed_data[ex.extractor.y == cls].iloc[:n_samples].index)
            fig_column = feature['dim_1_uni_transform'] if column else feature['dim_1']

            line_df = pd.DataFrame(columns=['timestamp', fig_column, 'class'])
            line_df[feature['dim_1_uni_transform']] = np.stack(ex.extractor.transformed_data \
                                                                   [SEPARATOR.join(
                    [feature['dim_1'], feature['dim_1_uni_transform']])] \
                                                               .loc[index_to_plot].values)[:,
                                                      slice_start:slice_end].reshape(-1)
            line_df[feature['dim_1']] = np.stack(ex.extractor.transformed_data \
                                                     [SEPARATOR.join([feature['dim_1'], ex.identity])] \
                                                 .loc[index_to_plot].values)[:, slice_start:slice_end].reshape(-1)
            line_df['class'] = np.repeat(ex.extractor.y[index_to_plot], length)
            line_df['Timestamp'] = np.repeat(np.arange(slice_start, slice_end).reshape(1, length), num_lines,
                                             axis=0).reshape(-1)
            line_df['id'] = np.repeat(np.arange(num_lines), length)
            fig1 = px.line(line_df, x='Timestamp', y=fig_column, color='class', line_group='id')

            fig2 = px.line(line_df, x='Timestamp', y=feature['dim_1_uni_transform'], color='class', line_group='id')
            fig2.update_traces(opacity=alpha)
            if agg in list(SCATTER_FUNCTIONS.keys()):
                scatter_df = pd.DataFrame()
                scatter_df['Timestamp'] = line_df.groupby('id')[feature['dim_1_uni_transform']].agg(
                    SCATTER_FUNCTIONS[agg][1]).values + slice_start
                scatter_df['agg'] = line_df.groupby('id')[feature['dim_1_uni_transform']].agg(
                    SCATTER_FUNCTIONS[agg][0]).values
                scatter_df['class'] = line_df.groupby('id')['class'].first().apply(lambda x: agg + ' ' + x)
                scatter_fig = px.scatter(scatter_df, x='Timestamp', y='agg', color='class')
                fig2 = go.Figure(data=fig2.data + scatter_fig.data)
            elif agg in list(LINE_FUNCTIONS.keys()):
                scatter_df = pd.DataFrame()
                scatter_df['Timestamp'] = np.repeat([[int(feature['slice_start']), int(feature['slice_end'])]],
                                                    num_lines, axis=0).reshape(-1)
                scatter_df['agg'] = np.repeat(
                    line_df.groupby('id')[feature['dim_1_uni_transform']].agg(LINE_FUNCTIONS[agg]).values, 2)
                scatter_df['class'] = np.repeat(
                    line_df.groupby('id')['class'].first().apply(lambda x: agg + ' ' + x).values, 2)
                scatter_df['id'] = np.repeat(np.arange(num_lines), 2)
                scatter_fig = px.line(scatter_df, x='Timestamp', y='agg', color='class', markers=True, line_group='id')
                fig2 = go.Figure(data=fig2.data + scatter_fig.data)
            elif agg == 'sum':
                line_df['cumsum'] = np.cumsum(np.stack(ex.extractor.transformed_data[SEPARATOR.join \
                    ([feature['dim_1'], feature['dim_1_uni_transform']])] \
                                                       .loc[index_to_plot].values)[:, slice_start:slice_end],
                                              axis=1).reshape(-1)
                line_df['class'] = line_df['class'].apply(lambda x: x + ' sum')
                scatter_fig = px.line(line_df, x='Timestamp', y='cumsum', color='class', line_group='id', markers=True)
                fig2 = go.Figure(data=fig2.data + scatter_fig.data)
            elif agg == 'trend':
                to_val = np.stack(
                    ex.extractor.transformed_data[SEPARATOR.join([feature['dim_1'], feature['dim_1_uni_transform']])] \
                        .loc[index_to_plot].values)[:, slice_start:slice_end]
                line_df['trend'] = np.stack([np.polyval(np.polyfit(x=np.arange(slice_start, slice_end), y=tv, deg=1), \
                                                        np.arange(slice_start, slice_end)) for tv in to_val]).reshape(
                    -1)
                line_df['class'] = line_df['class'].apply(lambda x: x + ' trend')
                scatter_fig = px.line(line_df, x='Timestamp', y='trend', color='class', line_group='id', markers=True)
                fig2 = go.Figure(data=fig2.data + scatter_fig.data)

            value_df = pd.DataFrame()
            value_df[feature['name']] = ex.extractor.feature_vector[feature['name']]
            value_df['class'] = ex.extractor.y
            value_df = value_df.groupby('class').agg(list).reset_index()

            hist_data = value_df[feature['name']]
            group_labels = value_df['class']

            # Create distplot with custom bin_size
            fig3 = ff.create_distplot(hist_data, group_labels, bin_size=.2, curve_type='normal', show_rug=rug)

            #     fig1.update_layout(
            #    margin=dict(l=0, r=0, t=0, b=0)
            # )
            #     fig2.update_layout(
            #    margin=dict(l=0, r=0, t=0, b=0)
            # )
            #     fig3.update_layout(
            #    margin=dict(l=0, r=0, t=0, b=0)
            # )

            return fig1, fig2, fig3

        @self.app.callback(
            [Output("modal", "is_open"),
             Output("modal-fv", "children")],
            [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks"), Input("fig1", "children")],
            [State("modal", "is_open")],
        )
        def toggle_modal(n1, n2, children, is_open):
            if ctx.triggered_id == 'open-modal':
                return not is_open, children
            return False, []

    def __open_browser(self):
        webbrowser.open_new("http://localhost:{}".format(8050))

    def explore(self, open=True):
        if open:
            Timer(1, self.__open_browser).start()
        self.app.run_server(debug=False, port=8050)
