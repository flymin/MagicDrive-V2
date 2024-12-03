import os
import re
import mimetypes
import fire
from functools import lru_cache
from flask import Flask, jsonify, request, send_file
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
base_path = ''
executor = ThreadPoolExecutor(max_workers=8)  # Adjust the number of workers based on your system's capabilities
debug_mode = False


def debug_print(*args, **kwargs):
    if debug_mode:
        print(*args, **kwargs)


def is_match_exclude(s):
    pattern = r'^epoch\d+-global_step\d+$'
    return bool(re.match(pattern, s))


def extract_numbers_and_parts(s):
    parts = re.split(r'(\d+)', s)
    return [(int(part) if part.isdigit() else part) for part in parts]


def sort_directories(directories):
    return sorted(directories, key=lambda d: extract_numbers_and_parts(d['name']))


@app.route('/directories', methods=['GET'])
def get_directories():
    directories = []
    for root, dirs, files in os.walk(base_path):
        if root == base_path:
            directories = [{'name': d, 'path': os.path.abspath(os.path.join(root, d))} for d in dirs]
            break
    print(f"list base_dir done")
    directories = sort_directories(directories)
    print(f"Directories at base path: {directories}")
    tree = build_tree(directories)
    print(f"Directory tree: {tree}")
    return jsonify(tree)


def build_tree(directories):
    tree = []
    futures = {executor.submit(contains_media, directory['path']): directory for directory in directories}
    for future in as_completed(futures):
        directory = futures[future]
        result = future.result()
        debug_print(f"Directory: {directory['path']}, Contains media: {result}")
        if result:
            path = directory['path']
            children = [{'name': d, 'path': os.path.join(path, d)}
                        for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            directory['children'] = build_tree(children)
            tree.append(directory)
    tree = sort_directories(tree)
    return tree


@lru_cache(maxsize=None)
def contains_media(path):
    debug_print(f"Checking media in path: {path}")
    base_name = os.path.basename(path)
    if is_match_exclude(base_name):
        return False
    media_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp4', '.mkv', '.avi', '.mov'}
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in media_extensions):
                debug_print(f"Media file found: {file} in {root}")
                return True
    return False


@app.route('/media', methods=['GET'])
def get_media():
    path = request.args.get('path')
    page = request.args.get('page', None)
    if page is None:
        pass
    else:
        page = int(page)
        per_page = request.args.get('per_page', 9)
    media_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and (mime_type.startswith('image') or mime_type.startswith('video')):
                media_files.append({'path': file_path, 'type': mime_type.split('/')[0]})
        break
    total = len(media_files)
    if page is None:
        debug_print(f"Media files: {media_files}, Total: {total}")
        return jsonify({
            'media': media_files,
            'total': total,
        })
    start = (page - 1) * per_page
    end = start + per_page
    debug_print(f"Media files: {media_files[start:end]}, Total: {total}, Page: {page}")
    return jsonify({
        'media': media_files[start:end],
        'total': total,
        'page': page,
        'pages': (total + per_page - 1) // per_page
    })


@app.route('/file', methods=['GET'])
def get_file():
    path = request.args.get('path')
    debug_print(f"Sending file: {path}")
    return send_file(path)


@app.route('/', methods=['GET'])
def serve_index():
    debug_print("Serving index.html")
    return send_file('index.html')


def main(base_path_arg, debug=False):
    global base_path, debug_mode
    base_path = base_path_arg
    debug_mode = debug
    debug_print(f"Starting server with base path: {base_path} and debug mode: {debug_mode}")

    app.run(debug=debug_mode)
    print("got ctrl+c")
    executor.shutdown()


if __name__ == '__main__':
    fire.Fire(main)
