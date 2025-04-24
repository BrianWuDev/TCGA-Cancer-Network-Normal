import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
import os
import json
import webbrowser
import colorsys
import math

def create_web_network(input_file='data/normal.csv', central_node='GCH1', output_file='Network Normal/index.html'):
    """Create web-based interactive network visualization and save as HTML file"""
    print(f"Loading data from {input_file}...")
    
    try:
        # Read and filter data
        df = pd.read_csv(input_file)
        filtered_df = df[df['PCC'] >= 0.8].copy()
        type_column = 'Tumor'
        
        # Count genes for each tissue type
        tissue_types = filtered_df[type_column].unique()
        tissue_gene_counts = Counter(filtered_df[type_column])
        print(f"Found {len(tissue_types)} tissue types with {len(filtered_df)} genes")
        
        # Create graph
        G = nx.Graph()
        
        # Add central node
        G.add_node(central_node, node_type='central')
        
        # Create color map for tissues
        # Generate HSV colors then convert to RGB for better differentiation
        color_list = [
            "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#377eb8",
            "#ff7f00", "#ffff33", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
            "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#00ffff"
        ]
        
        tissue_colors = {}
        for i, tissue in enumerate(tissue_types):
            tissue_colors[tissue] = color_list[i % len(color_list)]
        
        # Set initial positions - circular distribution
        fixed_positions = {}
        
        # Central node position
        fixed_positions[central_node] = (0.0, 0.0)
        
        # Tissue nodes distributed around the circumference
        radius = 400
        golden_angle = math.pi * (3 - math.sqrt(5))  # Golden angle for more uniform distribution
        
        for i, tissue in enumerate(tissue_types):
            angle = i * golden_angle
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            fixed_positions[tissue] = (x, y)
            
            # Add tissue node
            G.add_node(tissue, node_type='tissue', gene_count=tissue_gene_counts[tissue])
            G.add_edge(central_node, tissue, weight=5.0)
            
        # Add gene nodes (limit to 150 per tissue to match original image)
        max_genes_per_tissue = 150
        genes_added = 0
        
        for tissue in tissue_types:
            # Get genes for this tissue, sorted by correlation
            tissue_genes = filtered_df[filtered_df[type_column] == tissue].sort_values('PCC', ascending=False)
            
            # Limit genes per tissue
            if len(tissue_genes) > max_genes_per_tissue:
                print(f"Limiting {tissue} genes from {len(tissue_genes)} to {max_genes_per_tissue}")
                tissue_genes = tissue_genes.head(max_genes_per_tissue)
            
            # Add selected genes to the graph
            for idx, row in tissue_genes.iterrows():
                gene = row['Gene Symbol']
                pcc = row['PCC']
                
                if gene not in G:
                    G.add_node(gene, node_type='gene', pcc=pcc, tissue=tissue)
                    genes_added += 1
                
                G.add_edge(gene, tissue, weight=pcc * 3)
                
                # Set gene node position - form clusters around tissue nodes
                # Use spiral/concentric circle layout instead of random distribution
                # This will make genes of the same tissue form compact clusters
                angle = (idx / len(tissue_genes)) * 2 * np.pi  # Spiral angle
                # Higher PCC genes closer to tissue node center
                distance = 50 + (1 - pcc) * 150  # Distance based on PCC, higher correlation = closer
                
                # Add cloud distribution effect - use Gaussian noise
                # Genes closer to tissue nodes have less offset, distant ones have more, creating cloud effect
                noise_scale = 0.3 + 0.6 * (1 - pcc)  # Higher PCC = less noise, lower PCC = more noise
                angle_noise = (np.random.normal(0, noise_scale) * 0.5)
                distance_noise = (np.random.normal(0, noise_scale) * 50)
                
                # Final position calculation with cloud effect
                cloud_angle = angle + angle_noise
                cloud_distance = distance + distance_noise
                
                x = fixed_positions[tissue][0] + cloud_distance * np.cos(cloud_angle)
                y = fixed_positions[tissue][1] + cloud_distance * np.sin(cloud_angle)
                fixed_positions[gene] = (x, y)
        
        print(f"Added {genes_added} gene nodes to graph")
        
        # Node and connection data
        nodes_data = []
        links_data = []
        
        # Add central node
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
        
        # Add tissue and gene nodes
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
        
        # Add connections
        for source, target in G.edges():
            links_data.append({
                "source": source,
                "target": target,
                "value": float(G.edges[(source, target)].get('weight', 1))
            })
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate HTML content
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
                    <strong>Instructions:</strong><br>
                    • Click and drag nodes to move them<br>
                    • Use mouse wheel to zoom in/out<br>
                    • Click and drag background to move the entire network<br>
                    • Hover over nodes to see details
                </div>
                <div class="controls">
                    <button id="reset-zoom">Reset View</button>
                    <button id="auto-cluster">Optimize Layout</button>
                </div>
                <button id="download-btn" class="download-btn">Download High-Res PNG</button>
            </div>
            
            <script>
                // Network data
                const nodesData = NODES_DATA_PLACEHOLDER;
                const linksData = LINKS_DATA_PLACEHOLDER;
                
                // Create node ID to object mapping
                const nodeMap = {};
                nodesData.forEach(node => {
                    nodeMap[node.id] = node;
                });
                
                // Get elements
                const container = document.getElementById('container');
                const canvas = document.getElementById('network-canvas');
                const ctx = canvas.getContext('2d');
                const legend = document.getElementById('legend');
                const tooltip = document.getElementById('tooltip');
                const downloadBtn = document.getElementById('download-btn');
                const resetZoomBtn = document.getElementById('reset-zoom');
                const autoClusterBtn = document.getElementById('auto-cluster');
                
                // Set canvas dimensions
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
                
                // Min and max zoom levels
                const MIN_SCALE = 0.1;
                const MAX_SCALE = 5;
                
                // Download high quality PNG
                function downloadHighQualityPNG() {
                    // Create larger offscreen canvas for high-res rendering
                    const offscreenCanvas = document.createElement('canvas');
                    const dpr = window.devicePixelRatio || 2; // Use device pixel ratio or default to 2x
                    const scaleFactor = 3; // Increased resolution
                    
                    offscreenCanvas.width = width * dpr * scaleFactor;
                    offscreenCanvas.height = height * dpr * scaleFactor;
                    const offscreenCtx = offscreenCanvas.getContext('2d');
                    
                    // Scale to accommodate higher resolution
                    offscreenCtx.scale(dpr * scaleFactor, dpr * scaleFactor);
                    
                    // Set white background
                    offscreenCtx.fillStyle = 'white';
                    offscreenCtx.fillRect(0, 0, width, height);
                    
                    // Save current transform for global elements
                    offscreenCtx.save();
                    
                    // Add title
                    offscreenCtx.font = 'bold 20px Arial';
                    offscreenCtx.textAlign = 'center';
                    offscreenCtx.fillStyle = 'black';
                    offscreenCtx.fillText('Gene Association Network Centered on GCH1 in normal tissue (PCC >= 0.8)', width/2, 30);
                    
                    // Draw legend to offscreen canvas
                    drawLegendToContext(offscreenCtx, width, height);
                    
                    // Restore transform state
                    offscreenCtx.restore();
                    
                    // Draw network to offscreen canvas
                    drawNetworkToContext(offscreenCtx, width, height);
                    
                    // Convert to data URL and trigger download
                    try {
                        const dataUrl = offscreenCanvas.toDataURL('image/png');
                        const downloadLink = document.createElement('a');
                        downloadLink.href = dataUrl;
                        downloadLink.download = 'gene_association_network.png';
                        document.body.appendChild(downloadLink);
                        downloadLink.click();
                        document.body.removeChild(downloadLink);
                    } catch (e) {
                        console.error('Failed to download image:', e);
                        alert('Failed to download image. Check console for more information.');
                    }
                }
                
                // Draw legend to context
                function drawLegendToContext(context, contextWidth, contextHeight) {
                    // Set legend position and style
                    const legendX = 30;
                    const legendY = 60;
                    const legendWidth = 220;
                    const lineHeight = 25;
                    
                    // Draw legend background
                    context.fillStyle = 'white';
                    context.strokeStyle = '#ddd';
                    context.lineWidth = 1;
                    
                    // Calculate legend height - central node + all tissue nodes
                    const legendItems = 1 + nodesData.filter(node => node.node_type === 'tissue').length;
                    const legendHeight = legendItems * lineHeight + 20;
                    
                    // Draw legend box
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
                    
                    // Draw central node legend item
                    const centralNode = nodesData.find(node => node.node_type === 'central');
                    if (centralNode) {
                        // Draw legend dot
                        context.beginPath();
                        context.arc(legendX + 15, legendY + 20, 8, 0, Math.PI * 2);
                        context.fillStyle = 'red';
                        context.fill();
                        context.strokeStyle = 'black';
                        context.lineWidth = 1;
                        context.stroke();
                        
                        // Draw text
                        context.font = '14px Arial';
                        context.textAlign = 'left';
                        context.fillStyle = 'black';
                        context.fillText('GCH1 (central)', legendX + 30, legendY + 24);
                    }
                    
                    // Draw tissue node legend items
                    const tissueNodes = nodesData.filter(node => node.node_type === 'tissue');
                    tissueNodes.forEach((node, index) => {
                        const y = legendY + 20 + (index + 1) * lineHeight;
                        
                        // Draw legend dot
                        context.beginPath();
                        context.arc(legendX + 15, y, 8, 0, Math.PI * 2);
                        context.fillStyle = node.color;
                        context.fill();
                        context.strokeStyle = 'black';
                        context.lineWidth = 0.5;
                        context.stroke();
                        
                        // Draw text
                        context.font = '12px Arial';
                        context.textAlign = 'left';
                        context.fillStyle = 'black';
                        context.fillText(`${node.name} (n=${node.gene_count})`, legendX + 30, y + 4);
                    });
                }
                
                // Draw network to context
                function drawNetworkToContext(context, contextWidth, contextHeight) {
                    // Save current transform state
                    context.save();
                    
                    // Set transform to match current view
                    context.translate(offsetX, offsetY);
                    context.scale(scale, scale);
                    
                    // Draw connections
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
                    
                    // Draw gene nodes
                    nodesData.filter(node => node.node_type === 'gene').forEach(node => {
                        context.beginPath();
                        context.arc(node.x, node.y, node.size, 0, Math.PI * 2);
                        context.fillStyle = node.color;
                        context.fill();
                    });
                    
                    // Draw tissue nodes
                    nodesData.filter(node => node.node_type === 'tissue').forEach(node => {
                        context.beginPath();
                        context.arc(node.x, node.y, node.size, 0, Math.PI * 2);
                        context.fillStyle = node.color;
                        context.fill();
                        context.strokeStyle = 'black';
                        context.lineWidth = 1;
                        context.stroke();
                    });
                    
                    // Draw central node
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
                    
                    // Draw labels
                    context.font = '12px Arial';
                    context.textAlign = 'center';
                    context.textBaseline = 'middle';
                    
                    // Labels for tissue and central nodes
                    nodesData.filter(node => node.node_type === 'tissue' || node.node_type === 'central').forEach(node => {
                        const x = node.x;
                        const y = node.y;
                        
                        // Measure text width
                        const textWidth = context.measureText(node.label).width;
                        const padding = 5;
                        const labelHeight = 16;
                        
                        // Draw background
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
                        
                        // Draw text
                        context.fillStyle = node.node_type === 'central' ? 'red' : 'black';
                        context.fillText(node.label, x, y);
                    });
                    
                    // Restore transform state
                    context.restore();
                }
                
                // Reset view
                function resetView() {
                    scale = 1;
                    offsetX = width / 2;
                    offsetY = height / 2;
                    draw();
                }
                
                // Optimize layout - further cluster genes by tissue
                function optimizeLayout() {
                    // Get all tissue nodes
                    const tissueNodes = nodesData.filter(node => node.node_type === 'tissue');
                    
                    // For each tissue, rearrange its genes
                    tissueNodes.forEach(tissue => {
                        const tissueGenes = nodesData.filter(node => 
                            node.node_type === 'gene' && node.tissue === tissue.name
                        );
                        
                        // Sort genes so higher PCC values are closer to tissue center
                        tissueGenes.sort((a, b) => b.pcc - a.pcc);
                        
                        // Use cloud layout
                        tissueGenes.forEach((gene, idx) => {
                            // Base angle and distance
                            const angle = (idx / tissueGenes.length) * 2 * Math.PI;
                            const baseDistance = 40 + (1 - gene.pcc) * 100;
                            
                            // Add randomness for cloud distribution
                            // Higher PCC values have less noise, creating a tighter core
                            const noiseScale = 0.2 + 0.5 * (1 - gene.pcc);
                            const angleNoise = (Math.random() - 0.5) * noiseScale * Math.PI;
                            const distanceNoise = (Math.random() * 2 - 1) * noiseScale * 60;
                            
                            // Apply cloud noise
                            const cloudAngle = angle + angleNoise;
                            const cloudDistance = Math.max(20, baseDistance + distanceNoise);
                            
                            // Update position
                            gene.x = tissue.x + cloudDistance * Math.cos(cloudAngle);
                            gene.y = tissue.y + cloudDistance * Math.sin(cloudAngle);
                        });
                    });
                    
                    // Redraw
                    draw();
                }
                
                // Resize canvas when window size changes
                function resizeCanvas() {
                    width = container.clientWidth;
                    height = container.clientHeight;
                    canvas.width = width;
                    canvas.height = height;
                    draw();
                }
                
                // Initialize
                function initialize() {
                    // Create legend
                    createLegend();
                    
                    // Set event listeners
                    canvas.addEventListener('mousedown', handleMouseDown);
                    canvas.addEventListener('mousemove', handleMouseMove);
                    window.addEventListener('mouseup', handleMouseUp);
                    canvas.addEventListener('wheel', handleWheel);
                    window.addEventListener('resize', resizeCanvas);
                    downloadBtn.addEventListener('click', downloadHighQualityPNG);
                    resetZoomBtn.addEventListener('click', resetView);
                    autoClusterBtn.addEventListener('click', optimizeLayout);
                    
                    // Set initial canvas size
                    resizeCanvas();
                    
                    // Apply cluster layout on initialization
                    optimizeLayout();
                }
                
                // Create legend
                function createLegend() {
                    // Add GCH1 central node
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
                    
                    // Add tissue nodes
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
                
                // Handle mouse down event
                function handleMouseDown(event) {
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = event.clientX - rect.left;
                    const mouseY = event.clientY - rect.top;
                    
                    // Check if clicked on a node
                    const node = getNodeAtPosition(mouseX, mouseY);
                    
                    if (node) {
                        // Drag node
                        dragNode = node;
                        dragOffsetX = (mouseX - offsetX) / scale - node.x;
                        dragOffsetY = (mouseY - offsetY) / scale - node.y;
                        canvas.style.cursor = 'grabbing';
                    } else {
                        // Drag canvas
                        isDraggingCanvas = true;
                        startX = mouseX;
                        startY = mouseY;
                        canvas.style.cursor = 'grabbing';
                    }
                }
                
                // Handle mouse move event
                function handleMouseMove(event) {
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = event.clientX - rect.left;
                    const mouseY = event.clientY - rect.top;
                    
                    if (dragNode) {
                        // Update dragged node position
                        dragNode.x = (mouseX - offsetX) / scale - dragOffsetX;
                        dragNode.y = (mouseY - offsetY) / scale - dragOffsetY;
                        draw();
                    } else if (isDraggingCanvas) {
                        // Pan canvas
                        offsetX += mouseX - startX;
                        offsetY += mouseY - startY;
                        startX = mouseX;
                        startY = mouseY;
                        draw();
                    } else {
                        // Check hover
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
                
                // Handle mouse up event
                function handleMouseUp() {
                    dragNode = null;
                    isDraggingCanvas = false;
                    canvas.style.cursor = 'default';
                }
                
                // Handle wheel event
                function handleWheel(event) {
                    event.preventDefault();
                    
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = event.clientX - rect.left;
                    const mouseY = event.clientY - rect.top;
                    
                    // Calculate mouse position relative to canvas (accounting for current offset and scale)
                    const x = (mouseX - offsetX) / scale;
                    const y = (mouseY - offsetY) / scale;
                    
                    // Update zoom
                    if (event.deltaY < 0) {
                        // Zoom in
                        scale *= 1.1;
                    } else {
                        // Zoom out
                        scale *= 0.9;
                    }
                    
                    // Constrain zoom range
                    scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
                    
                    // Adjust offset to keep mouse position at same point
                    offsetX = mouseX - x * scale;
                    offsetY = mouseY - y * scale;
                    
                    draw();
                }
                
                // Get node at position
                function getNodeAtPosition(x, y) {
                    // Check from smallest to largest nodes (by size) so smaller nodes can be selected even if behind larger ones
                    for (let i = nodesData.length - 1; i >= 0; i--) {
                        const node = nodesData[i];
                        const nodeScreenX = node.x * scale + offsetX;
                        const nodeScreenY = node.y * scale + offsetY;
                        const size = node.size * scale;
                        
                        // Check if click position is within node
                        const distance = Math.sqrt(Math.pow(x - nodeScreenX, 2) + Math.pow(y - nodeScreenY, 2));
                        if (distance <= size) {
                            return node;
                        }
                    }
                    return null;
                }
                
                // Show tooltip
                function showTooltip(node, clientX, clientY) {
                    let content = '';
                    
                    if (node.node_type === 'central') {
                        content = `<strong>${node.name}</strong><br>Central node`;
                    } else if (node.node_type === 'tissue') {
                        content = `<strong>${node.name}</strong><br>Gene count: ${node.gene_count}`;
                    } else if (node.node_type === 'gene') {
                        content = `<strong>${node.name}</strong><br>PCC: ${node.pcc.toFixed(3)}<br>Tissue: ${node.tissue}`;
                    }
                    
                    tooltip.innerHTML = content;
                    tooltip.style.left = (clientX + 10) + 'px';
                    tooltip.style.top = (clientY + 10) + 'px';
                    tooltip.style.display = 'block';
                }
                
                // Draw network
                function draw() {
                    // Clear canvas
                    ctx.clearRect(0, 0, width, height);
                    
                    // Use shared drawing function
                    drawNetworkToContext(ctx, width, height);
                }
                
                // Initialize app
                initialize();
            </script>
        </body>
        </html>
        '''
        
        # Replace data placeholders
        html_content = html_content.replace('NODES_DATA_PLACEHOLDER', json.dumps(nodes_data))
        html_content = html_content.replace('LINKS_DATA_PLACEHOLDER', json.dumps(links_data))
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Network visualization saved to {output_file} and opened in browser")
        
        # Open in browser
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
    
    create_web_network(input_file='data/normal.csv', output_file='Network Normal/index.html') 