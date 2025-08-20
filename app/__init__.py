# # app/__init__.py
#
# from flask import Flask
# from flask_cors import CORS
#
# import os
#
# def create_app():
#     app = Flask(__name__)
#     CORS(app)
#     app.config["UPLOAD_FOLDER"] = os.path.abspath("uploads")
#     from app.route import register_routes  # ✅ 改成匯入 register_routes 函數
#     # ✅ 註冊所有路由
#     register_routes(app)
#
#     return app
# app/__init__.py - 基本版本
# app/__init__.py - 加入數據載入測試

# from flask import Flask, jsonify, render_template
# from flask_cors import CORS
# import os
#
# def create_app():
#     app = Flask(
#         __name__,
#         template_folder="templates",       # ✅ HTML 存放位置
#         static_folder="static"            # ✅ CSS、JS 存放位置
#     )
#
#     CORS(app)
#
#     # 測試數據載入
#     try:
#         import pandas as pd
#         df = pd.read_excel("data/TESTData.xlsx")
#         print(f"✓ Successfully loaded Excel with {len(df)} rows")
#         data_status = f"Data loaded: {len(df)} rows"
#     except Exception as e:
#         print(f"✗ Error loading data: {e}")
#         data_status = f"Data load failed: {str(e)}"
#
#     # @app.route("/")
#     # def index():
#     #     return render_template("index.html")
#     @app.route("/")
#     def index():
#         try:
#             return render_template("index.html")
#         except Exception as e:
#             return f"HTML template rendering failed: {e}"
#
#     @app.route("/healthz")
#     def healthz():
#         return "ok", 200
#
#     # 導入並註冊其他路由（放在最後）
#     try:
#         from app.route import register_routes
#         register_routes(app)
#         print("✓ Additional routes registered")
#     except Exception as e:
#         print(f"✗ Error registering additional routes: {e}")
#
#     return app

# app/__init__.py - 除錯版本


# 可以測試LOG
# import os
# import sys
#
# # 最早的除錯輸出
# print("=== DEBUG: Starting app/__init__.py ===")
# print(f"Python version: {sys.version}")
# print(f"Current working directory: {os.getcwd()}")
#
# try:
#     from flask import Flask, jsonify
#     print("✓ Flask imported successfully")
# except Exception as e:
#     print(f"✗ Error importing Flask: {e}")
#     sys.exit(1)
#
# try:
#     from flask_cors import CORS
#     print("✓ Flask-CORS imported successfully")
# except Exception as e:
#     print(f"✗ Error importing Flask-CORS: {e}")
#
# def create_app():
#     print("=== DEBUG: create_app() called ===")
#
#     try:
#         app = Flask(__name__)
#         print("✓ Flask app created")
#     except Exception as e:
#         print(f"✗ Error creating Flask app: {e}")
#         raise
#
#     try:
#         CORS(app)
#         print("✓ CORS configured")
#     except Exception as e:
#         print(f"✗ Error configuring CORS: {e}")
#
#     # 基本路由（不涉及模板）
#     @app.route("/")
#     def index():
#         return "Hello from create_app! Service is running."
#
#     @app.route("/healthz")
#     def healthz():
#         return "ok", 200
#
#     @app.route("/debug-info")
#     def debug_info():
#         return {
#             "python_version": sys.version,
#             "cwd": os.getcwd(),
#             "files_in_cwd": os.listdir("."),
#             "app_folder_exists": os.path.exists("app"),
#             "templates_folder_exists": os.path.exists("app/templates"),
#             "static_folder_exists": os.path.exists("app/static")
#         }
#
#     # 測試數據載入（可能的問題來源）
#     try:
#         print("=== DEBUG: Attempting to load data ===")
#         import pandas as pd
#         print("✓ Pandas imported")
#
#         if os.path.exists("data/TESTData.xlsx"):
#             print("✓ Excel file exists")
#             df = pd.read_excel("data/TESTData.xlsx")
#             print(f"✓ Successfully loaded Excel with {len(df)} rows")
#             data_status = f"Data loaded: {len(df)} rows"
#         else:
#             print("✗ Excel file not found")
#             data_status = "Excel file not found"
#
#     except Exception as e:
#         print(f"✗ Error loading data: {e}")
#         data_status = f"Data load failed: {str(e)}"
#
#     @app.route("/data-status")
#     def data_status_route():
#         return {"status": data_status}
#
#     print("=== DEBUG: create_app() completed successfully ===")
#     return app
#
# print("=== DEBUG: app/__init__.py loaded successfully ===")

# app/__init__.py - 步驟1：加回模板功能
# import os
# import sys
#
# print("=== DEBUG: Starting app/__init__.py ===")
# print(f"Current working directory: {os.getcwd()}")
#
# try:
#     from flask import Flask, jsonify, render_template
#     print("✓ Flask and render_template imported successfully")
# except Exception as e:
#     print(f"✗ Error importing Flask: {e}")
#
# try:
#     from flask_cors import CORS
#     print("✓ Flask-CORS imported successfully")
# except Exception as e:
#     print(f"✗ Error importing Flask-CORS: {e}")
#
# def create_app():
#     print("=== DEBUG: create_app() called ===")
#
#     # 檢查模板路徑
#     print("=== DEBUG: Checking template paths ===")
#     template_paths = [
#         "templates",           # 相對於根目錄
#         "app/templates",       # 相對於根目錄的 app/templates
#         "./templates",         # 明確的相對路徑
#         "./app/templates"      # 明確的相對路徑
#     ]
#
#     template_folder = None
#     for path in template_paths:
#         if os.path.exists(path):
#             template_folder = path
#             print(f"✓ Found templates at: {path}")
#             try:
#                 files = os.listdir(path)
#                 print(f"  Files in {path}: {files}")
#                 if "index.html" in files:
#                     print("  ✓ index.html found!")
#                 else:
#                     print("  ✗ index.html not found!")
#             except Exception as e:
#                 print(f"  Error listing files: {e}")
#             break
#
#     if not template_folder:
#         print("✗ No template folder found!")
#         template_folder = "templates"  # 使用預設值
#
#     try:
#         app = Flask(__name__, template_folder=template_folder)
#         print(f"✓ Flask app created with template_folder: {template_folder}")
#     except Exception as e:
#         print(f"✗ Error creating Flask app: {e}")
#         raise
#
#     try:
#         CORS(app)
#         print("✓ CORS configured")
#     except Exception as e:
#         print(f"✗ Error configuring CORS: {e}")
#
#     # 測試模板路由
#     @app.route("/")
#     def index():
#         try:
#             print("=== DEBUG: Attempting to render template ===")
#             template_path = os.path.join(app.template_folder, "index.html")
#             print(f"Looking for template at: {template_path}")
#
#             if not os.path.exists(template_path):
#                 return f"Template not found at: {template_path}"
#
#             return render_template("index.html")
#         except Exception as e:
#             print(f"✗ Template rendering error: {e}")
#             return f"Template rendering failed: {str(e)}"
#
#     # 備用路由（不使用模板）
#     @app.route("/simple")
#     def simple():
#         return "Simple route without template works!"
#
#     @app.route("/healthz")
#     def healthz():
#         return "ok", 200
#
#     # 數據載入
#     try:
#         import pandas as pd
#         df = pd.read_excel("data/TESTData.xlsx")
#         print(f"✓ Successfully loaded Excel with {len(df)} rows")
#         data_status = f"Data loaded: {len(df)} rows"
#     except Exception as e:
#         print(f"✗ Error loading data: {e}")
#         data_status = f"Data load failed: {str(e)}"
#     @app.route("/debug")
#     def debug():
#         info = {
#             "cwd": os.getcwd(),
#             "template_folder": app.template_folder,
#             "template_path_exists": os.path.exists(app.template_folder),
#             "files_in_template_folder": [],
#             "static_folder": app.static_folder,
#             "static_path_exists": os.path.exists(app.static_folder),
#             "files_in_static_folder": []
#         }
#         try:
#             info["files_in_template_folder"] = os.listdir(app.template_folder)
#         except Exception as e:
#             info["files_in_template_folder"] = [f"Error: {str(e)}"]
#
#         try:
#             info["files_in_static_folder"] = os.listdir(app.static_folder)
#         except Exception as e:
#             info["files_in_static_folder"] = [f"Error: {str(e)}"]
#
#         return info
#
#     @app.route("/data-status")
#     def data_status_route():
#         return {"status": data_status}
#
#     print("=== DEBUG: create_app() completed successfully ===")
#     return app
#
# print("=== DEBUG: app/__init__.py loaded successfully ===")

# app/__init__.py - 修正靜態文件配置
# import os
# import sys
#
# print("=== DEBUG: Starting app/__init__.py ===")
# print(f"Current working directory: {os.getcwd()}")
#
# try:
#     from flask import Flask, jsonify, render_template
#     print("✓ Flask and render_template imported successfully")
# except Exception as e:
#     print(f"✗ Error importing Flask: {e}")
#
# try:
#     from flask_cors import CORS
#     print("✓ Flask-CORS imported successfully")
# except Exception as e:
#     print(f"✗ Error importing Flask-CORS: {e}")
#
# def create_app():
#     print("=== DEBUG: create_app() called ===")
#
#     # 檢查模板和靜態文件路徑
#     print("=== DEBUG: Checking paths ===")
#
#     # 找到正確的路徑
#     template_folder = None
#     static_folder = None
#
#     # 檢查可能的路徑組合
#     path_combinations = [
#         ("templates", "static"),                    # 根目錄
#         ("app/templates", "app/static"),            # app 子目錄
#         ("./templates", "./static"),                # 明確相對路徑
#         ("./app/templates", "./app/static")         # 明確 app 相對路徑
#     ]
#
#     for template_path, static_path in path_combinations:
#         if os.path.exists(template_path) and os.path.exists(static_path):
#             template_folder = template_path
#             static_folder = static_path
#             print(f"✓ Found templates at: {template_path}")
#             print(f"✓ Found static at: {static_path}")
#
#             # 列出文件
#             try:
#                 template_files = os.listdir(template_path)
#                 static_files = os.listdir(static_path)
#                 print(f"  Template files: {template_files}")
#                 print(f"  Static files: {static_files}")
#             except Exception as e:
#                 print(f"  Error listing files: {e}")
#             break
#
#     # 如果找不到，使用預設值
#     if not template_folder:
#         template_folder = "app/templates"
#         print(f"✗ Using default template folder: {template_folder}")
#     if not static_folder:
#         static_folder = "app/static"
#         print(f"✗ Using default static folder: {static_folder}")
#
#     try:
#         # app = Flask(
#         #     __name__,
#         #     template_folder=template_folder,
#         #     static_folder=static_folder,
#         #     static_url_path='/static'  # 明確指定靜態文件 URL 路徑
#         # )
#         app = Flask(
#             __name__,
#             template_folder=template_folder,
#             static_folder="static"
#
#         )
#         print(f"✓ Flask app created")
#         print(f"  Template folder: {app.template_folder}")
#         print(f"  Static folder: {app.static_folder}")
#         print(f"  Static URL path: {app.static_url_path}")
#     except Exception as e:
#         print(f"✗ Error creating Flask app: {e}")
#         raise
#
#     try:
#         CORS(app)
#         print("✓ CORS configured")
#     except Exception as e:
#         print(f"✗ Error configuring CORS: {e}")
#
#     # 測試靜態文件路由
#     @app.route("/test-static")
#     def test_static():
#         try:
#             from flask import url_for
#             css_url = url_for('static', filename='index.css')
#             js_url = url_for('static', filename='index.js')
#             return f"CSS URL: {css_url}<br>JS URL: {js_url}"
#         except Exception as e:
#             return f"Static URL generation failed: {str(e)}"
#
#     # 測試模板路由（帶錯誤處理）
#     @app.route("/")
#     def index():
#         try:
#             print("=== DEBUG: Attempting to render template ===")
#
#             # 檢查模板文件存在
#             template_path = os.path.join(app.template_folder, "index.html")
#             print(f"Template path: {template_path}")
#             print(f"Template exists: {os.path.exists(template_path)}")
#
#             # 檢查靜態文件存在
#             css_path = os.path.join(app.static_folder, "index.css")
#             js_path = os.path.join(app.static_folder, "index.js")
#             print(f"CSS path: {css_path}, exists: {os.path.exists(css_path)}")
#             print(f"JS path: {js_path}, exists: {os.path.exists(js_path)}")
#
#             if not os.path.exists(template_path):
#                 return f"Template not found at: {template_path}"
#
#             return render_template("index.html")
#
#         except Exception as e:
#             print(f"✗ Template rendering error: {e}")
#             import traceback
#             traceback.print_exc()
#             return f"Template rendering failed: {str(e)}"
#
#     # 簡化版首頁（不依賴靜態文件）
#     @app.route("/simple-home")
#     def simple_home():
#         return """
#         <!DOCTYPE html>
#         <html lang="zh-Hant">
#         <head>
#             <meta charset="utf-8">
#             <title>Pill Detection - Simple</title>
#             <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         </head>
#         <body>
#             <div style="text-align: center; padding: 20px;">
#                 <h1>Medical Detection APP</h1>
#                 <p>簡化版本 - 不依賴外部CSS/JS</p>
#                 <p>服務正常運行中...</p>
#             </div>
#         </body>
#         </html>
#         """
#
#     @app.route("/healthz")
#     def healthz():
#         return "ok", 200
#
#     # 數據載入
#     try:
#         import pandas as pd
#         df = pd.read_excel("data/TESTData.xlsx")
#         print(f"✓ Successfully loaded Excel with {len(df)} rows")
#         data_status = f"Data loaded: {len(df)} rows"
#     except Exception as e:
#         print(f"✗ Error loading data: {e}")
#         data_status = f"Data load failed: {str(e)}"
#
#     @app.route("/debug")
#     def debug():
#         info = {
#             "cwd": os.getcwd(),
#             "template_folder": app.template_folder,
#             "template_path_exists": os.path.exists(app.template_folder),
#             "static_folder": app.static_folder,
#             "static_path_exists": os.path.exists(app.static_folder),
#             "data_status": data_status
#         }
#
#         try:
#             info["files_in_template_folder"] = os.listdir(app.template_folder)
#         except Exception as e:
#             info["files_in_template_folder"] = [f"Error: {str(e)}"]
#
#         try:
#             info["files_in_static_folder"] = os.listdir(app.static_folder)
#         except Exception as e:
#             info["files_in_static_folder"] = [f"Error: {str(e)}"]
#
#         return info
#
#     print("=== DEBUG: create_app() completed successfully ===")
#     return app
#
# print("=== DEBUG: app/__init__.py loaded successfully ===")

#服務啟動，但有問題
import os
import sys

print("=== DEBUG: Starting app/__init__.py ===")
print(f"Current working directory: {os.getcwd()}")

try:
    from flask import Flask, jsonify, render_template

    print("✓ Flask and render_template imported successfully")
except Exception as e:
    print(f"✗ Error importing Flask: {e}")

try:
    from flask_cors import CORS

    print("✓ Flask-CORS imported successfully")
except Exception as e:
    print(f"✗ Error importing Flask-CORS: {e}")


def create_app():
    print("=== DEBUG: create_app() called ===")

    # 檢查模板和靜態文件路徑
    print("=== DEBUG: Checking paths ===")

    # 找到正確的路徑
    template_folder = None
    static_folder = None

    # 檢查可能的路徑組合
    path_combinations = [
        ("templates", "static"),  # 根目錄
        ("app/templates", "app/static"),  # app 子目錄
        ("./templates", "./static"),  # 明確相對路徑
        ("./app/templates", "./app/static")  # 明確 app 相對路徑
    ]

    for template_path, static_path in path_combinations:
        if os.path.exists(template_path) and os.path.exists(static_path):
            template_folder = template_path
            static_folder = static_path
            print(f"✓ Found templates at: {template_path}")
            print(f"✓ Found static at: {static_path}")
            # 列出文件
            try:
                template_files = os.listdir(template_path)
                static_files = os.listdir(static_path)
                print(f"  Template files: {template_files}")
                print(f"  Static files: {static_files}")
            except Exception as e:
                print(f"  Error listing files: {e}")
            break

    # 如果找不到，使用預設值並確保目錄存在
    if not template_folder:
        template_folder = "app/templates"
        print(f"✗ Using default template folder: {template_folder}")
    if not static_folder:
        static_folder = "app/static"
        print(f"✗ Using default static folder: {static_folder}")

    # 確保目錄存在
    os.makedirs(template_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)

    try:
        app = Flask(
            __name__,
            template_folder=template_folder,
            static_folder=static_folder,
            static_url_path='/static'
        )
        print(f"✓ Flask app created")
        print(f"  Template folder: {app.template_folder}")
        print(f"  Static folder: {app.static_folder}")
        print(f"  Static URL path: {app.static_url_path}")
    except Exception as e:
        print(f"✗ Error creating Flask app: {e}")
        raise

    try:
        CORS(app)
        print("✓ CORS configured")
    except Exception as e:
        print(f"✗ Error configuring CORS: {e}")

    # 數據載入
    try:
        import pandas as pd
        df = pd.read_excel("data/TESTData.xlsx")
        print(f"✓ Successfully loaded Excel with {len(df)} rows")
        data_status = f"Data loaded: {len(df)} rows"
        # 將 df 設為 app 的屬性，讓其他模組可以使用
        app.df = df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        data_status = f"Data load failed: {str(e)}"
        app.df = None
    # 確保在註冊路由前匯入必要的模組
    from .route import register_routes
    # 註冊路由
    try:

        register_routes(app)
        print("✓ Routes registered successfully")
    except ImportError as e:
        print(f"✗ Error importing routes: {e}")
        # 如果無法匯入 routes，提供基本路由
        register_basic_routes(app, data_status)
    except Exception as e:
        print(f"✗ Error registering routes: {e}")
        register_basic_routes(app, data_status)

    print("=== DEBUG: create_app() completed successfully ===")
    return app


def register_basic_routes(app, data_status):
    """註冊基本路由作為備用"""
    print("✓ Registering basic routes as fallback")

    @app.route("/")
    def index():
        try:
            print("=== DEBUG: Attempting to render template ===")
            # 檢查模板文件存在
            template_path = os.path.join(app.template_folder, "index.html")
            print(f"Template path: {template_path}")
            print(f"Template exists: {os.path.exists(template_path)}")

            if not os.path.exists(template_path):
                # 如果模板不存在，創建一個簡單的模板
                create_default_template(app.template_folder)

            return render_template("index.html")
        except Exception as e:
            print(f"✗ Template rendering error: {e}")
            import traceback
            traceback.print_exc()
            return simple_home_html()

    @app.route("/simple-home")
    def simple_home():
        return simple_home_html()

    @app.route("/healthz")
    def healthz():
        return "ok", 200

    @app.route("/debug")
    def debug():
        info = {
            "cwd": os.getcwd(),
            "template_folder": app.template_folder,
            "template_path_exists": os.path.exists(app.template_folder),
            "static_folder": app.static_folder,
            "static_path_exists": os.path.exists(app.static_folder),
            "data_status": data_status
        }

        try:
            info["files_in_template_folder"] = os.listdir(app.template_folder)
        except Exception as e:
            info["files_in_template_folder"] = [f"Error: {str(e)}"]

        try:
            info["files_in_static_folder"] = os.listdir(app.static_folder)
        except Exception as e:
            info["files_in_static_folder"] = [f"Error: {str(e)}"]

        return info


def simple_home_html():
    """簡單的HTML首頁"""
    return """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="utf-8">
        <title>Pill Detection - Simple</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
            .container { max-width: 600px; margin: 0 auto; }
            h1 { color: #333; }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Medical Detection APP</h1>
            <div class="status">
                <p>簡化版本 - 服務正常運行中...</p>
                <p>如果您看到這個頁面，表示後端服務已啟動</p>
            </div>
            <p><a href="/debug">查看除錯資訊</a></p>
        </div>
    </body>
    </html>
    """


def create_default_template(template_folder):
    """創建預設的 index.html 模板"""
    template_content = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <title>Medical Detection APP</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='index.css') }}">
</head>
<body>
    <div class="container">
        <h1>Medical Detection APP</h1>
        <div class="upload-area">
            <p>藥物辨識系統</p>
            <input type="file" id="imageInput" accept="image/*">
            <div id="results"></div>
        </div>
    </div>
    <script src="{{ url_for('static', filename='index.js') }}"></script>
</body>
</html>"""

    try:
        template_path = os.path.join(template_folder, "index.html")
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        print(f"✓ Created default template at {template_path}")
    except Exception as e:
        print(f"✗ Error creating default template: {e}")


print("=== DEBUG: app/__init__.py loaded successfully ===")
