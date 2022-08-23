from flask_assets import Bundle
bundles = {

    'scss_all': Bundle(
        'app.scss', filters='pyscss', output='all.css'
    )
}