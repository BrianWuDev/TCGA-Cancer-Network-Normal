#  Gene Association Network Visualization

This repository presents an interactive web-based tool for visualizing a gene association network centered on **GCH1** in normal tissues. Genes with a high Pearson correlation coefficient (PCC ≥ 0.8) to GCH1 are displayed as nodes within a tissue-specific cluster layout.

**Web Application**  
Access the live visualization here:  
👉 [https://brianwudev.github.io/TCGA-Cancer-Network-Normal/](https://brianwudev.github.io/TCGA-Cancer-Network-Normal/)  
*(If the hyperlink fails, please copy and paste the above URL into your browser.)*

---

##  Features

- **Centralized network** focused on GCH1
- **Tissue-specific clustering** for co-expressed genes
- **Interactive controls**:
  - Click and drag nodes to reposition
  - Scroll wheel to zoom in and out
  - Drag background to pan across the network
- **Information on hover**: View details for each node
- **Layout optimization**: Automatically rearrange node positions
- **High-resolution PNG export** with title and legend

---

##  Use Case

This tool is designed to support biomedical researchers and bioinformaticians in:

- Identifying gene co-expression patterns
- Exploring potential regulatory or pathway interactions
- Preparing high-quality network figures for presentations or publications

---

##  Instructions

1. Open the web app using the link above.
2. **Navigate** the network:
   - Drag nodes to adjust the layout manually.
   - Use the mouse wheel to zoom.
   - Drag the background to move the full view.
3. **Hover** over a node to view its identifier and correlation score.
4. Use the **Optimize Layout** button to auto-adjust clusters.
5. Use the **Download PNG** button to export the current view.

---

##  Technical Implementation

- Developed using **HTML5 Canvas API**
- Purely client-side (no backend/server required)
- Optimized for Chrome and modern browsers
- Lightweight and suitable for embedding in other platforms

---

##  Example Visualization

> *(You may include a screenshot here using the following syntax:)*  
> `![Screenshot](images/screenshot.png)`

---

##  Citation & License

If you use this tool in academic work, please cite the corresponding publication (link TBD).  
Distributed under the [MIT License](LICENSE).

---

For questions, collaboration, or feedback, feel free to reach out via GitHub Issues or contact the author directly.
