import pandas as pd

from h2o_wave import Q, ui
from .utils import render_header, render_footer



"""Initialize client / browser"""
async def init_app(q:Q):

    q.client.df = pd.read_csv("./static/data.csv")
    q.client.plot_df = pd.read_csv("./static/ts_clusters.csv")
    # define layout and zones
    await init_layout(q)
    # render header
    await render_header(q)
    # render footer
    #await render_footer(q)


"""Layout definitions for the client"""
async def init_layout(q:Q) -> None: 
    q.page['meta'] = ui.meta_card(box='',
        #title='Time Series Clustering | H2O-3',
        layouts=[
            ui.layout(breakpoint='xs', zones=[
                #ui.zone(name='header'),
                ui.zone(name='main', size='100vh', zones=[
                    ui.zone(name='header', direction='row', size='120px'),
                    ui.zone(name='body', direction='row', size='1')
                ]),
                ui.zone(name='footer')
            ])
        ],
        theme='h2o-dark' # https://wave.h2o.ai/docs/color-theming
    )

    # # Header Card
    # q.page['header'] = ui.header_card(
    #     box=ui.box('header'), title = 'Time Series Clustering',
    #     #image='https://wave.h2o.ai/img/h2o-logo.svg', 
    #     secondary_items=[
    #         ui.tabs(name='tabs', value=f'#{q.args["#"]}' if q.args['#'] else '#home', link=True, items=[
    #             ui.tab(name="#home", label="Home", icon="Home"),
    #             ui.tab(name="#data", label="Data", icon="CommonDataServiceCDS"),
    #             ui.tab(name="#cluster", label="Models", icon="MachineLearning"),
    #         ]),
    #     ],
    #     items=[
    #         ui.persona(title='George Dantzig', subtitle='OG Data Scientist', size='xs'),
    #     ]
    # )
    

    