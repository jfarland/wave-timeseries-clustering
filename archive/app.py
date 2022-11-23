from h2o_wave import main, app, Q, data, on, handle_on, ui


from .initializers import init_app
from .utils import render_error_page


from .handlers import render_page


@app('/')
async def serve(q: Q):
    
    try:
        # init app if not already done so
        if not q.app.initialized:
            await init_app(q)
            q.app.initialized = True

        # render main page
        await render_page(q)

    # In the case of an exception, handle and report it to the user
    except Exception as err:
        await render_error_page(q, str(err))