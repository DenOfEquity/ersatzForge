import argparse
import json
import os
from modules.paths_internal import normalized_filepath, models_path, script_path, data_path, extensions_dir, extensions_builtin_dir  # noqa: F401
from pathlib import Path
from backend.args import parser

parser.add_argument("-f", action='store_true', help=argparse.SUPPRESS)  # allows running as root; implemented outside of webui
parser.add_argument("--update-all-extensions", action='store_true', help="launch.py argument: download updates for all extensions when starting the program")
parser.add_argument("--skip-python-version-check", action='store_true', help="launch.py argument: do not check python version")
parser.add_argument("--skip-torch-cuda-test", action='store_true', help="launch.py argument: do not check if CUDA is able to work properly")
parser.add_argument("--reinstall-xformers", action='store_true', help="launch.py argument: install the appropriate version of xformers even if you have some version already installed")
parser.add_argument("--reinstall-torch", action='store_true', help="launch.py argument: install the appropriate version of torch even if you have some version already installed")
parser.add_argument("--update-check", action='store_true', help="launch.py argument: check for updates at startup")
parser.add_argument("--test-server", action='store_true', help="launch.py argument: configure server for testing")
parser.add_argument("--log-startup", action='store_true', help="launch.py argument: print a detailed log of what's happening at startup")
parser.add_argument("--skip-prepare-environment", action='store_true', help="launch.py argument: skip all environment preparation")
parser.add_argument("--skip-google-blockly", action='store_true', help="launch.py argument: do not initialize google blockly modules")
parser.add_argument("--skip-install", action='store_true', help="launch.py argument: skip installation of packages")
parser.add_argument("--dump-sysinfo", action='store_true', help="launch.py argument: dump limited sysinfo file (without information about extensions, options) to disk and quit")
parser.add_argument("--data-dir", type=normalized_filepath, default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))), help="base path where all user data is stored")
parser.add_argument("--models-dir", type=normalized_filepath, default=None, help="base path where models are stored; overrides --data-dir")
parser.add_argument("--ckpt-dir", type=normalized_filepath, default=None, help="Path to directory with stable diffusion checkpoints")
parser.add_argument("--vae-dir", type=normalized_filepath, default=None, help="Path to directory with VAE files")
parser.add_argument("--text-encoder-dir", type=normalized_filepath, default=None, help="Path to directory with text encoder models")
parser.add_argument("--embeddings-dir", type=normalized_filepath, default=os.path.join(models_path, 'embeddings'), help="embeddings directory for textual inversion")
parser.add_argument("--hypernetwork-dir", type=normalized_filepath, default=os.path.join(models_path, 'hypernetworks'), help="hypernetwork directory")
parser.add_argument("--localizations-dir", type=normalized_filepath, default=os.path.join(script_path, 'localizations'), help="localizations directory")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "half", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
parser.add_argument("--ngrok", type=str, help="ngrok authtoken, alternative to gradio --share", default=None)
parser.add_argument("--ngrok-options", type=json.loads, help='The options to pass to ngrok in JSON format, e.g.: \'{"authtoken_from_env":true, "basic_auth":"user:password", "oauth_provider":"google", "oauth_allow_emails":"user@asdf.com"}\'', default=dict())
parser.add_argument("--enable-insecure-extension-access", action='store_true', help="enable extensions tab regardless of other options")
parser.add_argument("--codeformer-models-path", type=normalized_filepath, help="Path to directory with codeformer model file(s).", default=os.path.join(models_path, 'Codeformer'))
parser.add_argument("--gfpgan-models-path", type=normalized_filepath, help="Path to directory with GFPGAN model file(s).", default=os.path.join(models_path, 'GFPGAN'))
parser.add_argument("--clip-models-path", type=normalized_filepath, help="Path to directory with CLIP model file(s), for Interrogate options.", default=None)
parser.add_argument("--xformers", action='store_true', help="enable xformers for cross attention layers")
parser.add_argument("--use-ipex", action="store_true", help="use Intel XPU as torch device")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)
parser.add_argument("--ui-config-file", type=str, help="filename to use for ui configuration", default=os.path.join(data_path, 'ui-config.json'))
parser.add_argument("--hide-ui-dir-config", action='store_true', help="hide directory configuration from webui", default=False)
parser.add_argument("--freeze-settings", action='store_true', help="disable editing of all settings globally", default=False)
parser.add_argument("--freeze-settings-in-sections", type=str, help='disable editing settings in specific sections of the settings page by specifying a comma-delimited list such like "saving-images,upscaling". The list of setting names can be found in the modules/shared_options.py file', default=None)
parser.add_argument("--freeze-specific-settings", type=str, help='disable editing of individual settings by specifying a comma-delimited list like "samples_save,samples_format". The list of setting names can be found in the config.json file', default=None)
parser.add_argument("--ui-settings-file", type=str, help="filename to use for ui settings", default=os.path.join(data_path, 'config.json'))
parser.add_argument("--gradio-debug",  action='store_true', help="launch gradio with --debug option")
parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--gradio-auth-path", type=normalized_filepath, help='set gradio authentication file path ex. "/path/to/auth/file" same auth format as --gradio-auth', default=None)
parser.add_argument("--gradio-allowed-path", action='append', help="add path to gradio's allowed_paths, make it possible to serve files from it", default=[data_path])
parser.add_argument("--styles-file", type=str, action='append', help="path or wildcard path of styles files, allow multiple entries.", default=[])
parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
parser.add_argument("--disable-console-progressbars", action='store_true', help="do not output progressbars to console", default=False)
parser.add_argument("--api", action='store_true', help="use api=True to launch the API together with the webui (use --nowebui instead for only the API)")
parser.add_argument("--api-auth", type=str, help='Set authentication for API like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--api-log", action='store_true', help="use api-log=True to enable logging of all API requests")
parser.add_argument("--nowebui", action='store_true', help="use api=True to launch the API instead of the webui")
parser.add_argument("--ui-debug-mode", action='store_true', help="Don't load model to quickly launch UI")
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)", default=None)
parser.add_argument("--cors-allow-origins", type=str, help="Allowed CORS origin(s) in the form of a comma-separated list (no spaces)", default=None)
parser.add_argument("--cors-allow-origins-regex", type=str, help="Allowed CORS origin(s) in the form of a single regular expression", default=None)
parser.add_argument("--tls-keyfile", type=str, help="Partially enables TLS, requires --tls-certfile to fully function", default=None)
parser.add_argument("--tls-certfile", type=str, help="Partially enables TLS, requires --tls-keyfile to fully function", default=None)
parser.add_argument("--disable-tls-verify", action="store_false", help="When passed, enables the use of self-signed certificates.", default=None)
parser.add_argument("--server-name", type=str, help="Sets hostname of server", default=None)
parser.add_argument("--no-gradio-queue", action='store_true', help="Disables gradio queue; causes the webpage to use http requests instead of websockets; was the default in earlier versions")
parser.add_argument("--skip-version-check", action='store_true', help="Do not check versions of torch and xformers")
parser.add_argument("--no-hashing", action='store_true', help="disable sha256 hashing of checkpoints to help loading performance", default=False)
parser.add_argument('--subpath', type=str, help='customize the subpath for gradio, use with reverse proxy')
parser.add_argument('--api-server-stop', action='store_true', help='enable server stop/restart/kill via api')
parser.add_argument('--timeout-keep-alive', type=int, default=30, help='set timeout_keep_alive for uvicorn')
parser.add_argument("--disable-all-extensions", action='store_true', help="prevent all extensions from running regardless of any other settings", default=False)
parser.add_argument("--disable-extra-extensions", action='store_true', help="prevent all extensions except built-in from running regardless of any other settings", default=False)
parser.add_argument("--unix-filenames-sanitization", action='store_true', help="allow any symbols except '/' in filenames. May conflict with your browser and file system")
parser.add_argument("--filenames-max-length", type=int, default=128, help='maximal length of filenames of saved images. If you override it, it can conflict with your file system')
parser.add_argument("--no-prompt-history", action='store_true', help="disable read prompt from last generation feature; settings this argument will not create '--data_path/params.txt' file")

parser.add_argument("--disable-safe-unpickle", action="store_true", help="unused, but may be checked by extensions", default=False)

# Arguments added by forge.
parser.add_argument(
    '--forge-ref-a1111-home',
    type=Path,
    help="Look for models in an existing A1111 checkout's path",
    default=None
)
parser.add_argument(
    "--controlnet-dir",
    type=Path,
    help="Path to directory with ControlNet models",
    default=None,
)
parser.add_argument(
    "--controlnet-preprocessor-models-dir",
    type=Path,
    help="Path to directory with annotator model directories",
    default=None,
)
