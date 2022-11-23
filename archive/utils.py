
from h2o_wave import Q, ui

async def render_header(q: Q):

    q.page['header'] = ui.header_card(box='header', 
                                      color='card',
                                      title='Time Series Clustering', subtitle='H2O-3',
                                      image='https://wave.h2o.ai/img/h2o-logo.svg',
                                      items=[ui.persona(title=q.auth.username, 
                                                        subtitle='H2O AI Cloud User', 
                                                        size='xs', 
                                                        image=None, 
                                                        initials_color='Black')],
                                      )

async def render_footer(q: Q):

    # Footer Card
    q.page['footer'] = ui.footer_card(
        box=ui.box('footer'), caption="(c) 2022 H2O.ai. All rights reserved.", 
        items = [
            ui.inline(justify="end", items = [
                ui.links(label = "About Us", width='200px', items = [
                    ui.link(label="AI 4 Good", path='https://h2o.ai/company/ai-4-good/', target="_blank"),
                    ui.link(label="AI 4 Conservation", path="https://h2o.ai/company/ai-4-conservation/", target="_blank"),
                    ui.link(label="Democratize AI", path="https://h2o.ai/company/democratize-ai/", target="_blank"),
                    ui.link(label="Community", path="https://h2o.ai/community/", target="_blank")
                ]), 
                ui.links(label = "AI Engines", width='200px', items = [
                    ui.link(label="Driverless AI", path='https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/', target="_blank"),
                    ui.link(label="Open Source", path="https://h2o.ai/platform/ai-cloud/make/h2o/", target="_blank"),
                    ui.link(label="Computer Vision", path="https://h2o.ai/platform/ai-cloud/make/hydrogen-torch/", target="_blank"),
                    ui.link(label="Document AI", path="https://h2o.ai/platform/ai-cloud/make/document-ai/", target="_blank")
                ])
            ])
        ]
    )



async def render_error_page(q:Q, err:str):
    q.page['body'] = ui.form_card('body', items=[ui.text(err)])
