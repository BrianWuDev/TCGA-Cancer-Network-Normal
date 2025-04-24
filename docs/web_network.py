import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
import os
import json
import webbrowser
import colorsys
import math

def create_web_network(input_file='data/normal.csv', central_node='GCH1', output_file='network_visualization.html'):
    """Create web-based interactive network visualization and save as HTML file"""
    print(f"Loading data from {input_file}...")
    
    try:
        # 讀取並篩選數據
        df = pd.read_csv(input_file)
        filtered_df = df[df['PCC'] >= 0.8].copy()
        type_column = 'Tumor'
        
        # 計算每種組織類型的基因數量
        tissue_types = filtered_df[type_column].unique()
        tissue_gene_counts = Counter(filtered_df[type_column])
        print(f"Found {len(tissue_types)} tissue types with {len(filtered_df)} genes")
        
        # 創建圖形
        G = nx.Graph()
        
        # 添加中心節點
        G.add_node(central_node, node_type='central')
        
        # 創建組織顏色映射
        # 為了確保顏色有足夠的差異性，生成HSV顏色然後轉換為RGB
        color_list = [
            "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#377eb8",
            "#ff7f00", "#ffff33", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
            "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#00ffff"
        ]
        
        tissue_colors = {}
        for i, tissue in enumerate(tissue_types):
            tissue_colors[tissue] = color_list[i % len(color_list)]
        
        # 設置初始位置 - 圓形分布
        fixed_positions = {}
        
        # 中心節點位置
        fixed_positions[central_node] = (0.0, 0.0)
        
        # 組織節點分布在圓周上 - 類似圖片中的圓形分布
        radius = 400
        golden_angle = math.pi * (3 - math.sqrt(5))  # 黃金角，用於更均勻分布
        
        for i, tissue in enumerate(tissue_types):
            angle = i * golden_angle
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            fixed_positions[tissue] = (x, y)
            
            # 添加組織節點
            G.add_node(tissue, node_type='tissue', gene_count=tissue_gene_counts[tissue])
            G.add_edge(central_node, tissue, weight=5.0)
            
        # 添加基因節點（每個組織限制最多150個，以匹配原始圖像的視覺效果）
        max_genes_per_tissue = 150
        genes_added = 0
        
        for tissue in tissue_types:
            # 獲取該組織的基因，按相關性排序
            tissue_genes = filtered_df[filtered_df[type_column] == tissue].sort_values('PCC', ascending=False)
            
            # 限制每個組織的基因數量
            if len(tissue_genes) > max_genes_per_tissue:
                print(f"Limiting {tissue} genes from {len(tissue_genes)} to {max_genes_per_tissue}")
                tissue_genes = tissue_genes.head(max_genes_per_tissue)
            
            # 將選擇的基因添加到圖中
            for idx, row in tissue_genes.iterrows():
                gene = row['Gene Symbol']
                pcc = row['PCC']
                
                if gene not in G:
                    G.add_node(gene, node_type='gene', pcc=pcc, tissue=tissue)
                    genes_added += 1
                
                G.add_edge(gene, tissue, weight=pcc * 3)
                
                # 設置基因節點位置 - 圍繞組織節點分布，但形成集群
                # 使用螺旋/同心圓布局而非隨機分布
                # 這將使同一組織的基因形成緊湊的集群
                angle = (idx / len(tissue_genes)) * 2 * np.pi  # 螺旋角度
                # PCC越高的基因越靠近組織節點中心
                distance = 50 + (1 - pcc) * 150  # 基於PCC的距離調整，高相關性的基因更靠近組織節點
                
                # 添加雲狀分布效果 - 使用高斯噪聲偏移
                # 越接近組織節點的基因偏移越小，越遠的偏移越大，創造雲狀效果
                noise_scale = 0.3 + 0.6 * (1 - pcc)  # 高PCC值的基因噪聲小，低PCC值的噪聲大
                angle_noise = (np.random.normal(0, noise_scale) * 0.5)
                distance_noise = (np.random.normal(0, noise_scale) * 50)
                
                # 最終位置計算添加雲狀效果
                cloud_angle = angle + angle_noise
                cloud_distance = distance + distance_noise
                
                x = fixed_positions[tissue][0] + cloud_distance * np.cos(cloud_angle)
                y = fixed_positions[tissue][1] + cloud_distance * np.sin(cloud_angle)
                fixed_positions[gene] = (x, y)
        
        print(f"Added {genes_added} gene nodes to graph")
        
        # 節點與連接數據
        nodes_data = []
        links_data = []
        
        # 添加中心節點
        nodes_data.append({
            "id": central_node,
            "name": central_node,
            "node_type": "central",
            "size": 50,
            "color": "red",
            "x": fixed_positions[central_node][0],
            "y": fixed_positions[central_node][1],
            "label": f"{central_node} (central)"
        })
        
        # 添加組織節點
        for node in G.nodes():
            if node == central_node:
                continue
                
            node_type = G.nodes[node].get('node_type')
            
            if node_type == 'tissue':
                nodes_data.append({
                    "id": node,
                    "name": node,
                    "node_type": "tissue",
                    "size": 25,
                    "color": tissue_colors.get(node, "#999999"),
                    "gene_count": G.nodes[node].get('gene_count', 0),
                    "x": fixed_positions[node][0],
                    "y": fixed_positions[node][1],
                    "label": f"{node} (n={G.nodes[node].get('gene_count', 0)})"
                })
            elif node_type == 'gene':
                tissue = G.nodes[node].get('tissue')
                nodes_data.append({
                    "id": node,
                    "name": node,
                    "node_type": "gene",
                    "size": 3,
                    "color": tissue_colors.get(tissue, "#999999"),
                    "pcc": float(G.nodes[node].get('pcc', 0)),
                    "tissue": tissue,
                    "x": fixed_positions[node][0],
                    "y": fixed_positions[node][1]
                })
        
        # 添加連接
        for source, target in G.edges():
            links_data.append({
                "source": source,
                "target": target,
                "value": float(G.edges[(source, target)].get('weight', 1))
            })
            
        # 創建輸出目錄（如果不存在）
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成簡單的HTML內容
        html_content = '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Gene Association Network</title>
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                    font-family: Arial, sans-serif;
                }
                #container {
                    width: 100vw;
                    height: 100vh;
                    position: relative;
                    background-color: white;
                }
                #network-canvas {
                    position: absolute;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                }
                #title {
                    text-align: center;
                    font-size: 20px;
                    margin-top: 10px;
                    position: absolute;
                    width: 100%;
                    z-index: 10;
                }
                #legend {
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    max-height: 80vh;
                    overflow-y: auto;
                    z-index: 1000;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                    font-size: 12px;
                }
                .legend-color {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    display: inline-block;
                    margin-right: 5px;
                }
                .label {
                    background-color: white;
                    border: 1px solid rgba(0,0,0,0.2);
                    padding: 3px 6px;
                    border-radius: 3px;
                    font-size: 12px;
                    pointer-events: none;
                    white-space: nowrap;
                }
                .node-central {
                    cursor: pointer;
                    stroke: #000;
                    stroke-width: 1.5px;
                }
                .node-tissue {
                    cursor: pointer;
                    stroke: #000;
                    stroke-width: 1px;
                }
                .node-gene {
                    cursor: pointer;
                }
                .link {
                    stroke: #999;
                    stroke-opacity: 0.2;
                }
                #tooltip {
                    position: absolute;
                    background: rgba(255, 255, 255, 0.9);
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                    pointer-events: none;
                    font-size: 12px;
                    display: none;
                    z-index: 1000;
                }
                #instructions {
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    background: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 12px;
                    z-index: 1000;
                }
                .download-btn {
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 10px 15px;
                    font-size: 14px;
                    cursor: pointer;
                    z-index: 1000;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    transition: all 0.3s;
                }
                .download-btn:hover {
                    background: #45a049;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                }
                .controls {
                    position: absolute;
                    top: 50px;
                    right: 10px;
                    background: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    z-index: 1000;
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                }
                .controls button {
                    background: #f0f0f0;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    padding: 5px 10px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .controls button:hover {
                    background: #e0e0e0;
                }
            </style>
        </head>
        <body>
            <div id="container">
                <div id="title">Gene Association Network Centered on GCH1 in normal tissue (PCC >= 0.8)</div>
                <canvas id="network-canvas"></canvas>
                <div id="legend"></div>
                <div id="tooltip"></div>
                <div id="instructions">
                    <strong>操作說明：</strong><br>
                    • 點擊並拖動節點以移動位置<br>
                    • 滾輪放大縮小<br>
                    • 按住鼠標拖動背景移動整個網絡<br>
                    • 將鼠標懸停在節點上查看詳細信息
                </div>
                <div class="controls">
                    <button id="reset-zoom">重置視圖</button>
                    <button id="auto-cluster">優化布局</button>
                </div>
                <button id="download-btn" class="download-btn">下載高畫質PNG</button>
            </div>
            
            <script>
                // 網絡數據
                const nodesData = NODES_DATA_PLACEHOLDER;
                const linksData = LINKS_DATA_PLACEHOLDER;
                
                // 創建節點ID到對象的映射
                const nodeMap = {};
                nodesData.forEach(node => {
                    nodeMap[node.id] = node;
                });
                
                // 獲取元素
                const container = document.getElementById('container');
                const canvas = document.getElementById('network-canvas');
                const ctx = canvas.getContext('2d');
                const legend = document.getElementById('legend');
                const tooltip = document.getElementById('tooltip');
                const downloadBtn = document.getElementById('download-btn');
                const resetZoomBtn = document.getElementById('reset-zoom');
                const autoClusterBtn = document.getElementById('auto-cluster');
                
                // 設置畫布尺寸
                let width = container.clientWidth;
                let height = container.clientHeight;
                let scale = 1;
                let offsetX = width / 2;
                let offsetY = height / 2;
                let dragNode = null;
                let dragOffsetX = 0;
                let dragOffsetY = 0;
                let isDraggingCanvas = false;
                let startX = 0;
                let startY = 0;
                
                // 最小和最大縮放級別
                const MIN_SCALE = 0.1;
                const MAX_SCALE = 5;
                
                // 下載高品質PNG圖像
                function downloadHighQualityPNG() {
                    // 創建一個更大的離屏畫布，用於高解析度渲染
                    const offscreenCanvas = document.createElement('canvas');
                    const dpr = window.devicePixelRatio || 2; // 使用設備像素比或2倍分辨率
                    const scaleFactor = 3; // 提高解析度
                    
                    offscreenCanvas.width = width * dpr * scaleFactor;
                    offscreenCanvas.height = height * dpr * scaleFactor;
                    const offscreenCtx = offscreenCanvas.getContext('2d');
                    
                    // 縮放以適應更高的解析度
                    offscreenCtx.scale(dpr * scaleFactor, dpr * scaleFactor);
                    
                    // 設置白色背景
                    offscreenCtx.fillStyle = 'white';
                    offscreenCtx.fillRect(0, 0, width, height);
                    
                    // 保存當前變換以便繪製全局元素
                    offscreenCtx.save();
                    
                    // 添加標題
                    offscreenCtx.font = 'bold 20px Arial';
                    offscreenCtx.textAlign = 'center';
                    offscreenCtx.fillStyle = 'black';
                    offscreenCtx.fillText('Gene Association Network Centered on GCH1 in normal tissue (PCC >= 0.8)', width/2, 30);
                    
                    // 繪製圖例到離屏畫布 - 在右上角
                    drawLegendToContext(offscreenCtx, width, height);
                    
                    // 恢復變換狀態
                    offscreenCtx.restore();
                    
                    // 繪製網絡到離屏畫布
                    drawNetworkToContext(offscreenCtx, width, height);
                    
                    // 轉換為數據URL並觸發下載
                    try {
                        const dataUrl = offscreenCanvas.toDataURL('image/png');
                        const downloadLink = document.createElement('a');
                        downloadLink.href = dataUrl;
                        downloadLink.download = 'gene_association_network.png';
                        document.body.appendChild(downloadLink);
                        downloadLink.click();
                        document.body.removeChild(downloadLink);
                    } catch (e) {
                        console.error('下載圖像失敗:', e);
                        alert('下載圖像失敗，請檢查控制台獲取更多信息');
                    }
                }
                
                // 繪製圖例到指定的上下文
                function drawLegendToContext(context, contextWidth, contextHeight) {
                    // 設置圖例位置和樣式
                    const legendX = 30;
                    const legendY = 60;
                    const legendWidth = 220;
                    const lineHeight = 25;
                    
                    // 繪製圖例背景
                    context.fillStyle = 'white';
                    context.strokeStyle = '#ddd';
                    context.lineWidth = 1;
                    
                    // 計算圖例高度 - 中心節點 + 所有組織節點
                    const legendItems = 1 + nodesData.filter(node => node.node_type === 'tissue').length;
                    const legendHeight = legendItems * lineHeight + 20;
                    
                    // 繪製圖例框
                    context.beginPath();
                    context.roundRect(
                        legendX, 
                        legendY, 
                        legendWidth, 
                        legendHeight, 
                        5
                    );
                    context.fill();
                    context.stroke();
                    
                    // 繪製中心節點圖例項
                    const centralNode = nodesData.find(node => node.node_type === 'central');
                    if (centralNode) {
                        // 繪製圖例圓點
                        context.beginPath();
                        context.arc(legendX + 15, legendY + 20, 8, 0, Math.PI * 2);
                        context.fillStyle = 'red';
                        context.fill();
                        context.strokeStyle = 'black';
                        context.lineWidth = 1;
                        context.stroke();
                        
                        // 繪製文字
                        context.font = '14px Arial';
                        context.textAlign = 'left';
                        context.fillStyle = 'black';
                        context.fillText('GCH1 (central)', legendX + 30, legendY + 24);
                    }
                    
                    // 繪製組織節點圖例項
                    const tissueNodes = nodesData.filter(node => node.node_type === 'tissue');
                    tissueNodes.forEach((node, index) => {
                        const y = legendY + 20 + (index + 1) * lineHeight;
                        
                        // 繪製圖例圓點
                        context.beginPath();
                        context.arc(legendX + 15, y, 8, 0, Math.PI * 2);
                        context.fillStyle = node.color;
                        context.fill();
                        context.strokeStyle = 'black';
                        context.lineWidth = 0.5;
                        context.stroke();
                        
                        // 繪製文字
                        context.font = '12px Arial';
                        context.textAlign = 'left';
                        context.fillStyle = 'black';
                        context.fillText(`${node.name} (n=${node.gene_count})`, legendX + 30, y + 4);
                    });
                }
                
                // 在指定上下文繪製網絡
                function drawNetworkToContext(context, contextWidth, contextHeight) {
                    // 保存當前變換狀態
                    context.save();
                    
                    // 設置變換以匹配當前視圖
                    context.translate(offsetX, offsetY);
                    context.scale(scale, scale);
                    
                    // 繪製連接線
                    context.lineWidth = 0.5;
                    context.strokeStyle = 'rgba(150, 150, 150, 0.2)';
                    
                    linksData.forEach(link => {
                        const source = nodeMap[link.source];
                        const target = nodeMap[link.target];
                        
                        if (source && target) {
                            context.beginPath();
                            context.moveTo(source.x, source.y);
                            context.lineTo(target.x, target.y);
                            context.stroke();
                        }
                    });
                    
                    // 繪製基因節點
                    nodesData.filter(node => node.node_type === 'gene').forEach(node => {
                        context.beginPath();
                        context.arc(node.x, node.y, node.size, 0, Math.PI * 2);
                        context.fillStyle = node.color;
                        context.fill();
                    });
                    
                    // 繪製組織節點
                    nodesData.filter(node => node.node_type === 'tissue').forEach(node => {
                        context.beginPath();
                        context.arc(node.x, node.y, node.size, 0, Math.PI * 2);
                        context.fillStyle = node.color;
                        context.fill();
                        context.strokeStyle = 'black';
                        context.lineWidth = 1;
                        context.stroke();
                    });
                    
                    // 繪製中心節點
                    const centralNode = nodesData.find(node => node.node_type === 'central');
                    if (centralNode) {
                        context.beginPath();
                        context.arc(centralNode.x, centralNode.y, centralNode.size, 0, Math.PI * 2);
                        context.fillStyle = centralNode.color;
                        context.fill();
                        context.strokeStyle = 'black';
                        context.lineWidth = 2;
                        context.stroke();
                    }
                    
                    // 繪製標籤
                    context.font = '12px Arial';
                    context.textAlign = 'center';
                    context.textBaseline = 'middle';
                    
                    // 組織節點和中心節點的標籤
                    nodesData.filter(node => node.node_type === 'tissue' || node.node_type === 'central').forEach(node => {
                        const x = node.x;
                        const y = node.y;
                        
                        // 測量文本寬度
                        const textWidth = context.measureText(node.label).width;
                        const padding = 5;
                        const labelHeight = 16;
                        
                        // 繪製背景
                        context.fillStyle = 'white';
                        context.strokeStyle = '#aaa';
                        context.lineWidth = 1;
                        context.beginPath();
                        context.roundRect(
                            x - textWidth/2 - padding,
                            y - labelHeight/2 - padding,
                            textWidth + padding * 2,
                            labelHeight + padding * 2,
                            3
                        );
                        context.fill();
                        context.stroke();
                        
                        // 繪製文本
                        context.fillStyle = node.node_type === 'central' ? 'red' : 'black';
                        context.fillText(node.label, x, y);
                    });
                    
                    // 恢復變換狀態
                    context.restore();
                }
                
                // 重置視圖
                function resetView() {
                    scale = 1;
                    offsetX = width / 2;
                    offsetY = height / 2;
                    draw();
                }
                
                // 優化布局 - 進一步聚集相同組織的基因
                function optimizeLayout() {
                    // 獲取所有組織節點
                    const tissueNodes = nodesData.filter(node => node.node_type === 'tissue');
                    
                    // 針對每個組織，重新布局其基因
                    tissueNodes.forEach(tissue => {
                        const tissueGenes = nodesData.filter(node => 
                            node.node_type === 'gene' && node.tissue === tissue.name
                        );
                        
                        // 排序基因，使PCC更高的更靠近組織中心
                        tissueGenes.sort((a, b) => b.pcc - a.pcc);
                        
                        // 使用雲狀布局
                        tissueGenes.forEach((gene, idx) => {
                            // 基礎角度和距離
                            const angle = (idx / tissueGenes.length) * 2 * Math.PI;
                            const baseDistance = 40 + (1 - gene.pcc) * 100;
                            
                            // 添加隨機性實現雲狀分布
                            // PCC 值高的基因會更靠近組織中心，噪聲也更小
                            const noiseScale = 0.2 + 0.5 * (1 - gene.pcc);
                            const angleNoise = (Math.random() - 0.5) * noiseScale * Math.PI;
                            const distanceNoise = (Math.random() * 2 - 1) * noiseScale * 60;
                            
                            // 應用雲狀噪聲
                            const cloudAngle = angle + angleNoise;
                            const cloudDistance = Math.max(20, baseDistance + distanceNoise);
                            
                            // 更新位置
                            gene.x = tissue.x + cloudDistance * Math.cos(cloudAngle);
                            gene.y = tissue.y + cloudDistance * Math.sin(cloudAngle);
                        });
                    });
                    
                    // 重新繪製
                    draw();
                }
                
                // 當窗口大小改變時調整畫布大小
                function resizeCanvas() {
                    width = container.clientWidth;
                    height = container.clientHeight;
                    canvas.width = width;
                    canvas.height = height;
                    draw();
                }
                
                // 初始化
                function initialize() {
                    // 創建圖例
                    createLegend();
                    
                    // 設置事件監聽器
                    canvas.addEventListener('mousedown', handleMouseDown);
                    canvas.addEventListener('mousemove', handleMouseMove);
                    window.addEventListener('mouseup', handleMouseUp);
                    canvas.addEventListener('wheel', handleWheel);
                    window.addEventListener('resize', resizeCanvas);
                    downloadBtn.addEventListener('click', downloadHighQualityPNG);
                    resetZoomBtn.addEventListener('click', resetView);
                    autoClusterBtn.addEventListener('click', optimizeLayout);
                    
                    // 設置初始畫布大小
                    resizeCanvas();
                    
                    // 初始化時自動應用集群布局
                    optimizeLayout();
                }
                
                // 創建圖例
                function createLegend() {
                    // 添加GCH1中心節點
                    const centralNode = nodeMap['GCH1'];
                    if (centralNode) {
                        const centralItem = document.createElement('div');
                        centralItem.className = 'legend-item';
                        centralItem.innerHTML = `
                            <span class="legend-color" style="background-color: red;"></span>
                            <span>GCH1 (central)</span>
                        `;
                        legend.appendChild(centralItem);
                    }
                    
                    // 添加組織節點
                    const tissueNodes = nodesData.filter(node => node.node_type === 'tissue');
                    tissueNodes.forEach(node => {
                        const item = document.createElement('div');
                        item.className = 'legend-item';
                        item.innerHTML = `
                            <span class="legend-color" style="background-color: ${node.color};"></span>
                            <span>${node.name} (n=${node.gene_count})</span>
                        `;
                        legend.appendChild(item);
                    });
                }
                
                // 處理滑鼠按下事件
                function handleMouseDown(event) {
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = event.clientX - rect.left;
                    const mouseY = event.clientY - rect.top;
                    
                    // 檢查是否點擊了節點
                    const node = getNodeAtPosition(mouseX, mouseY);
                    
                    if (node) {
                        // 拖動節點
                        dragNode = node;
                        dragOffsetX = (mouseX - offsetX) / scale - node.x;
                        dragOffsetY = (mouseY - offsetY) / scale - node.y;
                        canvas.style.cursor = 'grabbing';
                    } else {
                        // 拖動畫布
                        isDraggingCanvas = true;
                        startX = mouseX;
                        startY = mouseY;
                        canvas.style.cursor = 'grabbing';
                    }
                }
                
                // 處理滑鼠移動事件
                function handleMouseMove(event) {
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = event.clientX - rect.left;
                    const mouseY = event.clientY - rect.top;
                    
                    if (dragNode) {
                        // 更新拖動的節點位置
                        dragNode.x = (mouseX - offsetX) / scale - dragOffsetX;
                        dragNode.y = (mouseY - offsetY) / scale - dragOffsetY;
                        draw();
                    } else if (isDraggingCanvas) {
                        // 平移畫布
                        offsetX += mouseX - startX;
                        offsetY += mouseY - startY;
                        startX = mouseX;
                        startY = mouseY;
                        draw();
                    } else {
                        // 檢查滑鼠懸停
                        const node = getNodeAtPosition(mouseX, mouseY);
                        if (node) {
                            canvas.style.cursor = 'pointer';
                            showTooltip(node, event.clientX, event.clientY);
                        } else {
                            canvas.style.cursor = 'default';
                            tooltip.style.display = 'none';
                        }
                    }
                }
                
                // 處理滑鼠釋放事件
                function handleMouseUp() {
                    dragNode = null;
                    isDraggingCanvas = false;
                    canvas.style.cursor = 'default';
                }
                
                // 處理滾輪事件
                function handleWheel(event) {
                    event.preventDefault();
                    
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = event.clientX - rect.left;
                    const mouseY = event.clientY - rect.top;
                    
                    // 計算鼠標相對於畫布的坐標（考慮當前偏移和縮放）
                    const x = (mouseX - offsetX) / scale;
                    const y = (mouseY - offsetY) / scale;
                    
                    // 更新縮放
                    if (event.deltaY < 0) {
                        // 放大
                        scale *= 1.1;
                    } else {
                        // 縮小
                        scale *= 0.9;
                    }
                    
                    // 限制縮放範圍
                    scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
                    
                    // 調整偏移量，使鼠標位置保持在同一點上
                    offsetX = mouseX - x * scale;
                    offsetY = mouseY - y * scale;
                    
                    draw();
                }
                
                // 在給定位置獲取節點
                function getNodeAtPosition(x, y) {
                    // 從大到小（按節點大小）檢查，以便先檢查較小的節點
                    for (let i = nodesData.length - 1; i >= 0; i--) {
                        const node = nodesData[i];
                        const nodeScreenX = node.x * scale + offsetX;
                        const nodeScreenY = node.y * scale + offsetY;
                        const size = node.size * scale;
                        
                        // 檢查點擊位置是否在節點內
                        const distance = Math.sqrt(Math.pow(x - nodeScreenX, 2) + Math.pow(y - nodeScreenY, 2));
                        if (distance <= size) {
                            return node;
                        }
                    }
                    return null;
                }
                
                // 顯示提示工具
                function showTooltip(node, clientX, clientY) {
                    let content = '';
                    
                    if (node.node_type === 'central') {
                        content = `<strong>${node.name}</strong><br>中心節點`;
                    } else if (node.node_type === 'tissue') {
                        content = `<strong>${node.name}</strong><br>基因數量: ${node.gene_count}`;
                    } else if (node.node_type === 'gene') {
                        content = `<strong>${node.name}</strong><br>PCC: ${node.pcc.toFixed(3)}<br>組織: ${node.tissue}`;
                    }
                    
                    tooltip.innerHTML = content;
                    tooltip.style.left = (clientX + 10) + 'px';
                    tooltip.style.top = (clientY + 10) + 'px';
                    tooltip.style.display = 'block';
                }
                
                // 繪製網絡
                function draw() {
                    // 清除畫布
                    ctx.clearRect(0, 0, width, height);
                    
                    // 使用通用繪製函數
                    drawNetworkToContext(ctx, width, height);
                }
                
                // 初始化應用
                initialize();
            </script>
        </body>
        </html>
        '''
        
        # 替換數據占位符
        html_content = html_content.replace('NODES_DATA_PLACEHOLDER', json.dumps(nodes_data))
        html_content = html_content.replace('LINKS_DATA_PLACEHOLDER', json.dumps(links_data))
        
        # 寫入HTML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Network visualization saved to {output_file} and opened in browser")
        
        # 在瀏覽器中打開
        webbrowser.open('file://' + os.path.abspath(output_file))
        
        return G, fixed_positions
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("Creating web-based interactive network visualization...")
    print("- Use browser controls to interact with the network")
    print("- The visualization will open automatically in your default browser")
    
    create_web_network(output_file='Network Analysis/web_network.html') 