import sys
import os
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from flask import Flask

# 1. SETUP PATHS
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "module1"))
sys.path.append(os.path.join(current_dir, "module2"))
sys.path.append(os.path.join(current_dir, "module3"))
sys.path.append(os.path.join(current_dir, "module4"))
sys.path.append(os.path.join(current_dir, "module6"))
sys.path.append(os.path.join(current_dir, "module7"))

# 2. IMPORT MODULES
from hub.app import app as hub_app
from module1.app import app as app1
from module2.app import app as app2
from module3.app import app as app3
from module4.app import app as app4
from module6.app import app as app6
from module7.app import app as app7

# 3. Map URLs to Apps
application = DispatcherMiddleware(hub_app, {
    '/module1': app1,
    '/module2': app2,
    '/module3': app3,
    '/module4': app4,
    '/module6': app6,
    '/module7': app7
})

if __name__ == '__main__':
    print(" Main Menu:   http://0.0.0.0:5050/")
    run_simple('0.0.0.0', 5050, application, use_reloader=True, use_debugger=True, threaded=True)
    #run_simple('localhost', 5050, application, use_reloader=True, use_debugger=True, threaded=True)