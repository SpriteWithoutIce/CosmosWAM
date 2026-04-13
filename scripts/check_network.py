#!/usr/bin/env python3
"""检测网络连接问题"""

import socket
import urllib.request
import os
import sys

print("="*60)
print("网络连接诊断")
print("="*60)

# 1. 检查代理
print("\n【1. 代理设置】")
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
found_proxy = False
for var in proxy_vars:
    val = os.environ.get(var)
    if val:
        print(f"  {var}={val}")
        found_proxy = True
if not found_proxy:
    print("  没有设置代理")

# 2. 检查 DNS
print("\n【2. DNS 解析】")
sites = ['api.swanlab.cn', 'wandb.ai', 'www.google.com', 'www.baidu.com']
for site in sites:
    try:
        ip = socket.gethostbyname(site)
        print(f"  {site} -> {ip} ✓")
    except Exception as e:
        print(f"  {site} -> 解析失败 ✗ ({e})")

# 3. 检查网络连通性
print("\n【3. 网络连通性】")
urls = [
    ('https://api.swanlab.cn', 'SwanLab API'),
    ('https://api.wandb.ai', 'Wandb API'),
    ('https://www.baidu.com', '百度'),
]

for url, name in urls:
    try:
        req = urllib.request.Request(url, method='HEAD')
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  {name}: HTTP {resp.status} ✓")
    except urllib.error.HTTPError as e:
        print(f"  {name}: HTTP {e.code} (可以连接但返回错误)")
    except Exception as e:
        print(f"  {name}: 连接失败 ✗ ({type(e).__name__})")

# 4. 检查是否需要代理
print("\n【4. 结论】")
print("  如果上面都失败，说明服务器无法直接访问外网")
print("  需要联系管理员开通网络或使用代理")
