import logging

import random

from h2o_wave import main, app, Q, ui, on, handle_on, data
from typing import Optional, List

import shutil

import pandas as pd
import os

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

from sklearn.mixture import GaussianMixture

import umap


# Use for page cards that should be removed when navigating away.
# For pages that should be always present on screen use q.page[key] = ...
def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card


# Remove all the cards related to navigation.
def clear_cards(q, ignore: Optional[List[str]] = []) -> None:
    if not q.client.cards:
        return

    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)


def table_from_df(
    df: pd.DataFrame,
    name: str,
    sortables: list = None,
    filterables: list = None,
    searchables: list = None,
    numerics: list = None,
    times: list = None,
    icons: dict = None,
    progresses: dict = None,
    min_widths: dict = None,
    max_widths: dict = None,
    link_col: str = None,
    multiple: bool = False,
    groupable: bool = False,
    downloadable: bool = False,
    resettable: bool = False,
    height: str = None,
    checkbox_visibility: str = None
) -> ui.table:
    """
    Convert a Pandas dataframe into Wave ui.table format.
    """

    if not sortables:
        sortables = []
    if not filterables:
        filterables = []
    if not searchables:
        searchables = []
    if not numerics:
        numerics = []
    if not times:
        times = []
    if not icons:
        icons = {}
    if not progresses:
        progresses = {}
    if not min_widths:
        min_widths = {}
    if not max_widths:
        max_widths = {}

    cell_types = {}
    for col in icons.keys():
        cell_types[col] = ui.icon_table_cell_type(color=icons[col]['color'])
    for col in progresses.keys():
        cell_types[col] = ui.progress_table_cell_type(color=progresses[col]['color'])

    columns = [ui.table_column(
        name=str(x),
        label=str(x),
        sortable=True if x in sortables else False,
        filterable=True if x in filterables else False,
        searchable=True if x in searchables else False,
        data_type='number' if x in numerics else ('time' if x in times else 'string'),
        cell_type=cell_types[x] if x in cell_types.keys() else None,
        min_width=min_widths[x] if x in min_widths.keys() else None,
        max_width=max_widths[x] if x in max_widths.keys() else None,
        link=True if x == link_col else False
    ) for x in df.columns.values]

    rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in df.iterrows()]

    table = ui.table(
        name=name,
        columns=columns,
        rows=rows,
        multiple=multiple,
        groupable=groupable,
        downloadable=downloadable,
        resettable=resettable,
        height=height,
        checkbox_visibility=checkbox_visibility
    )

    return table

@on('#home')
async def handle_home(q: Q):
    q.page['sidebar'].value = '#page1'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    add_card(q, 'article', ui.tall_article_preview_card(
        box=ui.box('vertical', height='1400px'), 
        title='Temporal Segmentation',
        subtitle='Click to begin...',
        name='#data',
        image='https://i.imgur.com/fxKvWXi.jpg',
        content='''
        '''
    ))

@on('#data')
async def handle_data(q: Q):
    q.page['sidebar'].value = '#data'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    if q.client.sample_df is not None:
        sample_df = q.client.sample_df[['Date', 'Weekly_Sales', 'Store', 'Dept']]
        sample_df['id'] = sample_df['Store'].astype(str) + "-" + sample_df['Dept'].astype(str)
        sample_df = sample_df[['Date', 'Weekly_Sales', 'id']]
        sample_df.columns = ['ds', 'y', 'id']
        sample_rows = {'Name':['M5 Competition'], 'Global Begin':[sample_df['ds'].min()], 'Global End':[sample_df['ds'].max()], 'Hierarchy': ['Store->Dept']}
        data_catalog = pd.DataFrame(sample_rows)
        #data_catalog = pd.concat([data_catalog, data_catalog], axis=0, ignore_index=True)
    else: 
        sample_df = pd.DataFrame()
        data_catalog = pd.DataFrame()

    data_catalog_columns = [
        ui.table_column(name='name',label='Name', link=True, sortable=True, filterable=True, searchable=True),
        ui.table_column(name='global_min', label='Global Min', sortable=True),
        ui.table_column(name='global_max', label='Global Max', sortable=True),
        ui.table_column(name='hierarchy', label='Hierarchy')
    ]
    #data_catalog_rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in data_catalog.iterrows()]
    data_catalog_rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in data_catalog.iterrows()]

    data_catalog_table = ui.table(name='data_catalog_table', columns=data_catalog_columns, rows=data_catalog_rows)

    add_card(q, 'data_catalog', ui.form_card(ui.box(zone="vertical", size="0"), items=[data_catalog_table]))
    
    #print(f'PRINTING CLIENT: {q.client}')
    if q.args.data_catalog_table is not None:
        print(f'PRINTING ARGS: {q.args.data_catalog_table}')

        ids = list(sample_df['id'].unique())

        if q.args.id_selector is not None:
            current_id =  q.args.id_selector
        else: 
            current_id = ids[0]

        add_card(q, 'time_series_selector', ui.form_card(
            box=ui.box(zone='vertical'),
            items = [
                ui.dropdown(name='id_selector',label='Select Time Series ID', trigger=True, value = current_id, choices=
                    [ui.choice(name=str(x), label=str(x)) for x in ids]
                ),
            ]
        ))

        colors = ['#3399ff', '#cc3300', '#00cc00', '#c00fff']

        # default to first ID
        local_df = sample_df[sample_df['id'] == current_id]

        ts_plot_rows = [tuple(x) for x in local_df.to_numpy()]

        # Create data buffer
        ts_plot_data = data('timestamp value id', rows = ts_plot_rows)

        # Reference: https://wave.h2o.ai/docs/examples/plot-line-groups
        #colors = "#3399ff #cc3300 #00cc00 #c00fff #ffff00"
        #colors = None
        add_card(q, 'timeseries_data_viz', ui.plot_card(
            box = ui.box('vertical', height = '1000px'), 
            title = 'Time Series Visualization',
            data = ts_plot_data,
            plot = ui.plot([
                ui.mark(
                    type='path', x='=timestamp', y='=value', color='=id',
                    y_title="Value", x_title='Timestamp', color_range=random.choice(colors))
            ])
        ))



        # TODO: create time series df head


    # IN PROGRESS: Accept File Uploads
    temp_folder_name = "./temp_data_for_file_upload_only"
    path = q.args.uploader
    if path:
        if not os.path.exists(temp_folder_name):
            os.mkdir(temp_folder_name)
        local_file_path = await q.site.download(url=path[0], path=temp_folder_name)
        new_df = pd.read_csv(local_file_path)
        # add_card(q, 'received', ui.markdown_card(box="inputs", title='Time Series Visualizer', content=str(new_df.iloc[0])))
        #if {"timestamp", "mean_kw", "humidity"}.issubset(new_df.columns):
            #q.client.full_df = pd.concat([q.client.full_df, new_df], axis=0).sort_values(['substation', 'timestamp'])
        #    pass
        shutil.rmtree(temp_folder_name)


    # Card for uploading
    add_card(q, 'upload_placeholder', ui.form_card(box=ui.box(zone="vertical", size="0"), items=[
        ui.text_xl('Upload a CSV'),
        ui.file_upload(name='uploader', label='Upload', multiple=False, file_extensions=['csv'])
        ]))

@on('#experiments')
async def handle_experiments(q: Q):
    q.page['sidebar'].value = '#experiments'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    if q.client.sample_df is not None:
        sample_df = q.client.sample_df[['Date', 'Weekly_Sales', 'Store', 'Dept']]
        sample_df['id'] = sample_df['Store'].astype(str) + "-" + sample_df['Dept'].astype(str)
        sample_df = sample_df[['Date', 'Weekly_Sales', 'id']]
        sample_df.columns = ['ds', 'y', 'id']
        sample_rows = {'Name':['M5 Competition'], 'Global Begin':[sample_df['ds'].min()], 'Global End':[sample_df['ds'].max()], 'Hierarchy': ['Store->Dept']}
        data_catalog = pd.DataFrame(sample_rows)
        #data_catalog = pd.concat([data_catalog, data_catalog], axis=0, ignore_index=True)
    else: 
        sample_df = pd.DataFrame()
        data_catalog = pd.DataFrame()

    data_catalog_columns = [
        ui.table_column(name='name',label='Name', link=True, sortable=True, filterable=True, searchable=True),
        ui.table_column(name='global_min', label='Global Min', sortable=True),
        ui.table_column(name='global_max', label='Global Max', sortable=True),
        ui.table_column(name='hierarchy', label='Hierarchy')
    ]
    #data_catalog_rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in data_catalog.iterrows()]
    data_catalog_rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in data_catalog.iterrows()]

    data_catalog_table = ui.table(name='data_catalog_table', columns=data_catalog_columns, rows=data_catalog_rows)

    add_card(q, 'data_catalog', ui.form_card(ui.box(zone="vertical", size="0"), items=[data_catalog_table]))
    
    #print(f'PRINTING CLIENT: {q.client}')
    if q.args.data_catalog_table is not None:
        print(f'PRINTING ARGS: {q.args.data_catalog_table}')

        feat_eng_choices = ['Efficient', 'All Possible', 'Minimal']
        reduce_dim = ['None', 'Manifold Learning', 'Principal Components', 't-sne', 'Deep Autoencoder']
        clustering_approaches = ['Gaussian Mixture Model', 'K-Means', 'Support Vector Machines']

        add_card(q, 'experiment_input', ui.form_card(
            box=ui.box(zone='vertical'),
            items = [
                ui.dropdown(name='engineered_features', label='Select Engineered Features', trigger=True, value = feat_eng_choices[0], choices=
                    [(ui.choice(x,x)) for x in feat_eng_choices]),
                ui.dropdown(name='reduce_dim',label='Perform Dimensionality Reduction', trigger=True, value = reduce_dim[0], choices=
                    [ui.choice(name=str(x), label=str(x)) for x in reduce_dim]
                ),
                ui.dropdown(name='clustering',label='Clustering Algorithm', trigger=True, value = clustering_approaches[0], choices=
                    [ui.choice(name=str(x), label=str(x)) for x in clustering_approaches]
                ),
                ui.button(name='launch_clustering', label='Launch', icon='HomeGroup')
            ]
        ))

        #settings = MinimalFCParameters()
        settings = EfficientFCParameters()

        extracted_features = extract_features(sample_df, column_id="id", column_sort="ds", default_fc_parameters=settings)
        extracted_features = extracted_features.dropna(axis=1)
        extracted_features.columns = extracted_features.columns.str.lstrip('y_')
        #extracted_features = extracted_features[extracted_features.columns[~extracted_features.isnull().all()]]
        print(extracted_features.head())

        embedding = umap.UMAP().fit_transform(extracted_features)
        dim = pd.DataFrame(embedding)
        dim.columns = ['umap1', 'umap2']
        #dim['label'] = labels

        model = GaussianMixture(n_components=2, covariance_type='full', reg_covar=0.1).fit(dim)
        dim['label'] = model.predict(dim)
        dim['label'] = dim['label'].astype(str)

        add_card(q, 'experiment_output', ui.form_card(
            box=ui.box('vertical'),
            items = [
                table_from_df(extracted_features.reset_index(), name='features', downloadable=True)
            ]
        ))

        #ts_plot_rows = [tuple(x) for x in local_df.to_numpy()]
        cluster_plot_data = data('umap1 umap2 label', rows =  [tuple(x) for x in dim.to_numpy()])

        add_card(q, 'cluster_viz', ui.plot_card(
            box = ui.box('vertical', height = '1000px'), 
            title = 'Time Series Visualization',
            data = cluster_plot_data,
            plot = ui.plot([
                ui.mark(
                    type='point', x='=umap1', y='=umap2', color='=label',
                    y_title="Projection 2", x_title='Project 1')
            ])
        ))


        # create label - id mapping and merge with original data
        dim['id'] = list(extracted_features.index)
        cluster_map = dim

        df_w_label = pd.merge(sample_df, cluster_map, how='left', on='id')

        # aggregate mean profiles by cluster
        df_agg = df_w_label.groupby(['ds', 'label'])['y'].mean().reset_index()

           #ts_plot_rows = [tuple(x) for x in local_df.to_numpy()]
        cluster_ts_plot_data = data('timestamp label value', rows =  [tuple(x) for x in df_agg.to_numpy()])

        add_card(q, 'cluster_ts_viz', ui.plot_card(
            box = ui.box('vertical', height = '1000px'), 
            title = 'Time Series Visualization',
            data = cluster_ts_plot_data,
            plot = ui.plot([
                ui.mark(
                    type='path', x='=timestamp', y='=value', color='=label',
                    y_title="value", x_title='time')
            ])
        ))

async def init(q: Q) -> None:
    q.page['meta'] = ui.meta_card(box='', layouts=[ui.layout(breakpoint='s', min_height='100vh', zones=[
        ui.zone('main', size='1', direction=ui.ZoneDirection.ROW, zones=[
            ui.zone('sidebar', size='250px'),
            ui.zone('body', zones=[
                ui.zone('content', zones=[
                    # Specify various zones and use the one that is currently needed. Empty zones are ignored.
                    ui.zone('horizontal', direction=ui.ZoneDirection.ROW),
                    ui.zone('vertical'),
                    ui.zone('grid', direction=ui.ZoneDirection.ROW, wrap='stretch', justify='center')
                ]),
            ]),
        ])
    ])],
    theme='h2o-dark' # https://wave.h2o.ai/docs/color-theming
    )
    q.page['sidebar'] = ui.nav_card(
        box='sidebar', color='primary', title='Time Series Clustering',
        value=f'#{q.args["#"]}' if q.args['#'] else '#home',
        image='https://wave.h2o.ai/img/h2o-logo.svg', items=[
            ui.nav_group('Menu', items=[
                ui.nav_item(name='#home', label='Home'),
                ui.nav_item(name='#data', label='Data'),
                ui.nav_item(name='#experiments', label='Experiments'),
                
            ]),
        ],
        secondary_items=[
            ui.persona(title='John Doe', subtitle='Developer', size='s',
                       image='https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&h=750&w=1260'),
        ]
    )


    # Load in sample dataset:
    #q.client.sample_df = pd.read_csv("s3://h2o-public-test-data/bigdata/server/energy-data-science/energy-df-sample.tar.gz")
    q.client.sample_df = pd.read_csv("s3://h2o-public-test-data/bigdata/server/walmart/walmart_tts_small_train.csv")


    # If no active hash present, render page1.
    if q.args['#'] is None:
        await handle_home(q)


@app('/')
async def serve(q: Q):
    # Run only once per client connection.
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True

    # Handle routing.
    await handle_on(q)
    await q.page.save()
