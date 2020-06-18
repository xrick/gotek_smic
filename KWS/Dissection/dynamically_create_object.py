
def model_factory_to_object(model_name):
    import glob
    avail_models = [model.replace('.py','').replace('Models/','') for model in glob.glob('Models/*.py') if 'init' not in model and 'basemodel' not in model]
    class_name = 'SanityModel'
    for model in avail_models:
        if model_name in model.lower():
            class_name = model
    import importlib
    model_ = getattr(importlib.import_module('Models.'+class_name), class_name)
    return model_