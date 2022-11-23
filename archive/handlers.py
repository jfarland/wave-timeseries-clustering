import pandas as pd

from typing import ItemsView
from h2o_wave import main, app, Q, data, on, handle_on, ui
from .utils import *

async def render_page(q: Q):

    # setup default view
    await render_clustering_page(q)

    # handle navigation clicks
    await handle_on(q)

    await q.page.save()

async def render_clustering_page(q):

    q.page['content'] = ui.form_card(box='body',
        items=[ui.text_l("Graphical View"),
               render_clusters_plot(q),
               ui.text_l("Table View"), 
               render_data_table(q),
               ui.text_l("Store Details"), 
               ui.text("Select a Store from the Table")
            ]
    )

    await q.page.save()


def render_clusters_plot(q):
    plot_data = q.client.plot_df

    plot_data = plot_data.transpose().reset_index()
    plot_data = plot_data.iloc[1::]
    plot_data.columns = ['Date'] + ["cluster: {}".format(i) for i in plot_data.columns[1::]]
    plot_data = pd.melt(plot_data, id_vars=['Date'], value_vars=['cluster: 0', 'cluster: 1', 'cluster: 2', 'cluster: 3'])

    viz = ui.visualization(
            plot=ui.plot([ui.mark(type='line', shape='circle', x='=Date', y='=value', color='=variable', y_min=0, x_title='Date', y_title='Weekly Sales')]),
            data=data(fields='Date variable value', pack=True, rows=[tuple(x) for x in plot_data.to_numpy()]),
        )

    return viz


def render_data_table(q):

    def get_data_type(data, col):
            if data.dtypes[col] == "object":
                return "string"
            else:
                return "number"

    def get_cell_type(row):
        if row['cluster'] == "1":
            return ["CLUSTER 1"]
        elif row['cluster'] == "2":
            return ["CLUSTER 2"]
        elif row['cluster'] == "3":
            return ["CLUSTER 3"]
        else:
            return ["CLUSTER 0"]

    commands = [ui.command(name='#details', label='Details', icon='Info')]

    df = q.client.df.head(n=50)
    df['cluster'] = df['cluster'].astype(int).astype(str)
    cols = ['Store'] + list(df.columns[-7::])
    tbl_columns = [ui.table_column(name=i, label=i, data_type=get_data_type(df, i), sortable=True) for i in cols]
    tbl_columns = [ui.table_column(name='tag', label='Cluster', filterable=True, cell_type=ui.tag_table_cell_type(name='tags',
                            tags=[
                                ui.tag(label='CLUSTER 0', color='$gray'),
                                ui.tag(label='CLUSTER 1', color='#D2E3F8'),
                                ui.tag(label='CLUSTER 2', color='$orange'),
                                ui.tag(label='CLUSTER 3', color='$mint'),
                            ]
                    ))] + tbl_columns + [ui.table_column(name='actions', label='Actions', cell_type=ui.menu_table_cell_type(name='commands', commands=commands))]

    tbl_rows = [ui.table_row(name='row{}'.format(i), cells=get_cell_type(df.iloc[i]) + [str(df.iloc[i][k]) for k in cols]) for i in range(30)]
    ts_table = ui.table(name='ts_table', 
                             columns=tbl_columns,
                             rows=tbl_rows,
                             )

    return ts_table

def render_details(q):
    plot_data = q.client.df
    plot_data = plot_data[plot_data.Store == 1]
    plot_data = plot_data[[i for i in plot_data.columns if i not in ['cluster', 'Store']]].transpose().reset_index()
    plot_data.columns = ['Date', 'Weekly_Sales']

    viz = ui.visualization(
            plot=ui.plot([ui.mark(type='line', shape='circle', x='=Date', y='=Weekly_Sales', y_min=0, x_title='Date', y_title='Weekly Sales')]),
            data=data(fields='Date Weekly_Sales', pack=True, rows=[tuple(x) for x in plot_data.to_numpy()]),
        )

    return viz



@on('#details')
async def show_reasons(q: Q):

    q.page['content'].items[5] = render_details(q)

    await q.page.save()