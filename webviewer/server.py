#!/usr/bin/env python3
"""Development server for the GGGS WebGL viewer with model browsing API.

Serves static webviewer files and provides API endpoints to list and load
model outputs directly from the server.

Usage:
    python webviewer/server.py [--port 8080] [--output_dir output]
"""

import argparse
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, unquote


class ViewerHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving webviewer static files + model browsing API."""

    output_dir = None  # Set by main()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == '/api/models':
            self._handle_list_models()
        elif path.startswith('/data/'):
            self._handle_data_file(path[6:])  # strip '/data/'
        else:
            super().do_GET()

    def _handle_list_models(self):
        """Scan output directory and return structured model listing."""
        output = self.output_dir
        if not os.path.isdir(output):
            self._json_response({'error': f'Output dir not found: {output}'}, 404)
            return

        models = []
        for entry in sorted(os.listdir(output)):
            model_path = os.path.join(output, entry)
            if not os.path.isdir(model_path):
                continue
            model = self._scan_model(model_path, entry)
            if model:
                models.append(model)
            else:
                # Check for nested scan directories (e.g., dtu_baseline/scan24)
                for sub in sorted(os.listdir(model_path)):
                    sub_path = os.path.join(model_path, sub)
                    if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, 'cfg_args')):
                        sub_model = self._scan_model(sub_path, f'{entry}/{sub}')
                        if sub_model:
                            models.append(sub_model)

        self._json_response(models)

    def _scan_model(self, model_path, name):
        """Scan a single model directory for available files."""
        cfg_path = os.path.join(model_path, 'cfg_args')
        if not os.path.exists(cfg_path):
            return None

        model = {
            'name': name,
            'path': os.path.relpath(model_path, self.output_dir),
            'has_cameras': os.path.exists(os.path.join(model_path, 'cameras.json')),
            'iterations': [],
        }

        # Parse cfg_args for key info
        try:
            with open(cfg_path) as f:
                cfg_text = f.read()
            from argparse import Namespace
            cfg = eval(cfg_text)
            model['config'] = {
                'sh_degree': cfg.sh_degree,
                'sg_degree': cfg.sg_degree,
                'resolution': cfg.resolution,
                'white_background': cfg.white_background,
                'kernel_size': cfg.kernel_size,
            }
        except Exception:
            model['config'] = {}

        # Scan point_cloud iterations
        pc_dir = os.path.join(model_path, 'point_cloud')
        if os.path.isdir(pc_dir):
            for iter_dir in sorted(os.listdir(pc_dir)):
                if not iter_dir.startswith('iteration_'):
                    continue
                iter_num = int(iter_dir.split('_')[1])
                iter_path = os.path.join(pc_dir, iter_dir)
                files = {}

                ply = os.path.join(iter_path, 'point_cloud.ply')
                if os.path.exists(ply):
                    files['ply'] = {
                        'url': f'/data/{model["path"]}/point_cloud/{iter_dir}/point_cloud.ply',
                        'size_mb': round(os.path.getsize(ply) / 1024 / 1024, 1),
                    }

                splat = os.path.join(iter_path, 'point_cloud.splat')
                if os.path.exists(splat):
                    files['splat'] = {
                        'url': f'/data/{model["path"]}/point_cloud/{iter_dir}/point_cloud.splat',
                        'size_mb': round(os.path.getsize(splat) / 1024 / 1024, 1),
                    }

                if files:
                    model['iterations'].append({
                        'iteration': iter_num,
                        'files': files,
                    })

        # cameras.json URL
        if model['has_cameras']:
            model['cameras_url'] = f'/data/{model["path"]}/cameras.json'

        return model

    def _handle_data_file(self, rel_path):
        """Serve a file from the output directory."""
        file_path = os.path.join(self.output_dir, rel_path)
        file_path = os.path.realpath(file_path)

        # Security: ensure path is within output_dir
        output_real = os.path.realpath(self.output_dir)
        if not file_path.startswith(output_real):
            self._json_response({'error': 'Access denied'}, 403)
            return

        if not os.path.isfile(file_path):
            self._json_response({'error': f'File not found: {rel_path}'}, 404)
            return

        # Determine content type
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.ply': 'application/octet-stream',
            '.splat': 'application/octet-stream',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
        }
        content_type = content_types.get(ext, 'application/octet-stream')

        file_size = os.path.getsize(file_path)
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(file_size))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        with open(file_path, 'rb') as f:
            # Stream in 1MB chunks for large files
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging: skip large file transfers
        msg = format % args
        if '/data/' in msg and ('200' in msg or '304' in msg):
            return
        sys.stderr.write(f'[{self.log_date_time_string()}] {msg}\n')


def main():
    parser = argparse.ArgumentParser(description='GGGS WebGL Viewer Server')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--output_dir', default=None,
                        help='Path to output directory (default: auto-detect)')
    args = parser.parse_args()

    # Find output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        # Auto-detect: look for 'output' relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'output')

    if not os.path.isdir(output_dir):
        print(f'Warning: output directory not found: {output_dir}')

    ViewerHandler.output_dir = output_dir

    # Serve from webviewer directory
    webviewer_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(webviewer_dir)

    server = HTTPServer((args.bind, args.port), ViewerHandler)
    print(f'GGGS Viewer Server')
    print(f'  Webviewer: {webviewer_dir}')
    print(f'  Output dir: {output_dir}')
    print(f'  http://{args.bind}:{args.port}')
    print(f'  API: http://{args.bind}:{args.port}/api/models')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down.')
        server.shutdown()


if __name__ == '__main__':
    main()
