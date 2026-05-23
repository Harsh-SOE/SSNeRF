import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, distance
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Plant Phenotyping Analysis (2.5D)", layout="wide", page_icon="🌱")

# Title
st.title("🌱 Plant Phenotyping Analysis Tool (2.5D Top-Down View)")
st.markdown("### Optimized for Top-Down RGB-D Camera Point Clouds")

def read_ply_file(filename):
    """Read PLY file and extract point coordinates"""
    with open(filename, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == 'end_header':
                break

        num_vertices = 0
        properties = []
        for line in header:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                properties.append(line.split()[2])

        format_type = [line for line in header if line.startswith('format')][0]

        if 'ascii' in format_type:
            data = []
            for _ in range(num_vertices):
                line = f.readline().decode('utf-8').strip()
                values = [float(x) for x in line.split()]
                data.append(values)
            return np.array(data), properties
        else:
            import struct
            data = []
            for _ in range(num_vertices):
                point = struct.unpack('fff', f.read(12))
                data.append(point)
            return np.array(data), properties

def identify_stem_center_2_5d(points):
    """
    Identify stem center for 2.5D top-down view
    Stem is at the center of the XY plane
    Returns: stem_center_xy (2D point in XY plane)
    """
    # Use points with highest Z values (closest to camera, likely stem/center)
    z_threshold = np.percentile(points[:, 2], 75)  # Top 25% in Z
    central_points = points[points[:, 2] >= z_threshold]

    if len(central_points) < 5:
        # Fallback: geometric center
        stem_center_xy = np.array([
            (points[:, 0].min() + points[:, 0].max()) / 2,
            (points[:, 1].min() + points[:, 1].max()) / 2
        ])
    else:
        # Center of mass of high points in XY
        stem_center_xy = np.array([
            central_points[:, 0].mean(),
            central_points[:, 1].mean()
        ])

    return stem_center_xy

def calculate_leaf_angles_2_5d(points, clusters, stem_center_xy):
    """
    Calculate leaf angles for 2.5D top-down view
    - Azimuthal angle: direction in XY plane from stem center
    - Tilt angle: angle of leaf surface from horizontal (XY plane)
    """
    unique_clusters = np.unique(clusters)
    leaf_angles = []

    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue

        cluster_points = points[clusters == cluster_id]

        if len(cluster_points) < 10:
            continue

        # Leaf centroid
        leaf_centroid = cluster_points.mean(axis=0)
        leaf_centroid_xy = leaf_centroid[:2]

        # 1. AZIMUTHAL ANGLE (direction from stem in XY plane)
        vector_from_stem = leaf_centroid_xy - stem_center_xy
        distance_from_stem = np.linalg.norm(vector_from_stem)

        # Angle in XY plane (0° = +X axis, counterclockwise)
        azimuthal_angle = np.degrees(np.arctan2(vector_from_stem[1], vector_from_stem[0]))
        if azimuthal_angle < 0:
            azimuthal_angle += 360

        # 2. TILT ANGLE (how much leaf tilts from horizontal XY plane)
        # Use PCA to find leaf plane normal
        pca = PCA(n_components=3)
        pca.fit(cluster_points)

        # The normal to the leaf plane is the 3rd component (least variance)
        leaf_normal = pca.components_[2]

        # Make sure normal points upward (positive Z component)
        if leaf_normal[2] < 0:
            leaf_normal = -leaf_normal

        # Angle between leaf normal and Z-axis (vertical from camera)
        # This tells us if leaf is horizontal (90°) or tilted toward/away from camera (0° or closer to 0°)
        z_axis = np.array([0, 0, 1])
        cos_angle = np.dot(leaf_normal, z_axis) / (np.linalg.norm(leaf_normal) * np.linalg.norm(z_axis))
        cos_angle = np.clip(cos_angle, -1, 1)

        tilt_from_horizontal = 90 - np.degrees(np.arccos(cos_angle))

        # 3. LEAF SPREAD (how far leaf extends from stem)
        # Calculate the main axis of the leaf in XY plane
        pca_xy = PCA(n_components=2)
        pca_xy.fit(cluster_points[:, :2])

        # Length along principal axis
        leaf_length = pca.explained_variance_[0] ** 0.5 * 4  # 2 std devs
        leaf_width = pca.explained_variance_[1] ** 0.5 * 4

        # Average Z (depth from camera)
        avg_depth = leaf_centroid[2]

        leaf_angles.append({
            'Cluster_ID': cluster_id,
            'Azimuthal_Angle': round(azimuthal_angle, 2),
            'Tilt_from_Horizontal': round(tilt_from_horizontal, 2),
            'Distance_from_Center': round(distance_from_stem, 2),
            'Leaf_Length': round(leaf_length, 2),
            'Leaf_Width': round(leaf_width, 2),
            'Avg_Depth': round(avg_depth, 2),
            'Center_X': round(leaf_centroid[0], 2),
            'Center_Y': round(leaf_centroid[1], 2),
            'Center_Z': round(leaf_centroid[2], 2),
            'Num_Points': len(cluster_points)
        })

    return pd.DataFrame(leaf_angles)

def create_3d_pointcloud_plot(points, clusters=None, stem_center=None, title="3D Point Cloud"):
    """Create interactive 3D point cloud visualization using Plotly"""

    traces = []

    # Main point cloud
    if clusters is not None:
        colors = clusters
        colorscale = 'Viridis'
    else:
        colors = points[:, 2]
        colorscale = 'Greens'

    trace_points = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        name='Point Cloud',
        marker=dict(
            size=2,
            color=colors,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Depth (Z)" if clusters is None else "Cluster")
        ),
        text=[f'X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}' 
              for x, y, z in points],
        hoverinfo='text'
    )
    traces.append(trace_points)

    # Add stem center marker if provided
    if stem_center is not None:
        z_avg = points[:, 2].mean()
        trace_stem = go.Scatter3d(
            x=[stem_center[0]],
            y=[stem_center[1]],
            z=[z_avg],
            mode='markers',
            name='Stem Center',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond'
            ),
            hoverinfo='name'
        )
        traces.append(trace_stem)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='X (Horizontal)',
            yaxis_title='Y (Horizontal)',
            zaxis_title='Z (Depth from Camera)',
            aspectmode='data'
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

def create_top_down_view(points, clusters=None, stem_center=None, leaf_angles_df=None):
    """Create 2D top-down view (XY plane)"""

    traces = []

    # Main point cloud in XY
    if clusters is not None:
        colors = clusters
        colorscale = 'Viridis'
        colorbar_title = "Cluster"
    else:
        colors = points[:, 2]
        colorscale = 'Greens'
        colorbar_title = "Depth"

    trace_points = go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode='markers',
        name='Plant (Top View)',
        marker=dict(
            size=4,
            color=colors,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=colorbar_title)
        ),
        text=[f'X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}' 
              for x, y, z in points],
        hoverinfo='text'
    )
    traces.append(trace_points)

    # Add stem center
    if stem_center is not None:
        trace_stem = go.Scatter(
            x=[stem_center[0]],
            y=[stem_center[1]],
            mode='markers',
            name='Stem Center',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            hoverinfo='name'
        )
        traces.append(trace_stem)

        # Add arrows showing leaf directions
        if leaf_angles_df is not None and len(leaf_angles_df) > 0:
            for _, leaf in leaf_angles_df.iterrows():
                # Draw arrow from stem to leaf
                angle_rad = np.radians(leaf['Azimuthal_Angle'])
                arrow_length = leaf['Distance_from_Center']

                end_x = stem_center[0] + arrow_length * np.cos(angle_rad)
                end_y = stem_center[1] + arrow_length * np.sin(angle_rad)

                trace_arrow = go.Scatter(
                    x=[stem_center[0], end_x],
                    y=[stem_center[1], end_y],
                    mode='lines',
                    name=f'Leaf {leaf["Cluster_ID"]}',
                    line=dict(width=2, color='orange'),
                    showlegend=False,
                    hovertext=f'Leaf {leaf["Cluster_ID"]}: {leaf["Azimuthal_Angle"]:.1f}°',
                    hoverinfo='text'
                )
                traces.append(trace_arrow)

    layout = go.Layout(
        title="Top-Down View (XY Plane) - Leaf Arrangement",
        xaxis_title='X (Horizontal)',
        yaxis_title='Y (Horizontal)',
        height=600,
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

def analyze_plant_phenotype_2_5d(points, eps=0.08, min_samples=10, min_leaf_points=20):
    """Analyze plant phenotypic traits from 2.5D top-down point cloud"""

    traits = {}

    # 1. CANOPY DIMENSIONS (primary measurement for top-down view)
    canopy_width_x = points[:, 0].max() - points[:, 0].min()
    canopy_width_y = points[:, 1].max() - points[:, 1].min()
    canopy_diameter = np.sqrt(canopy_width_x**2 + canopy_width_y**2)
    canopy_area_bbox = canopy_width_x * canopy_width_y

    traits['Canopy Width (X)'] = canopy_width_x
    traits['Canopy Width (Y)'] = canopy_width_y
    traits['Canopy Diameter'] = canopy_diameter
    traits['Canopy Area (BBox)'] = canopy_area_bbox

    # 2. DEPTH VARIATION (Z-axis)
    depth_range = points[:, 2].max() - points[:, 2].min()
    traits['Depth Range (Z)'] = depth_range

    # 3. PLANT VOLUME
    try:
        hull = ConvexHull(points)
        plant_volume = hull.volume
        traits['Plant Volume (Convex Hull)'] = plant_volume
        traits['Point Density'] = len(points) / plant_volume
    except:
        traits['Plant Volume (Convex Hull)'] = None
        traits['Point Density'] = None

    # 4. IDENTIFY STEM CENTER
    stem_center_xy = identify_stem_center_2_5d(points)

    # 5. LEAF SEGMENTATION using DBSCAN
    points_normalized = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(points_normalized)

    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters[unique_clusters != -1])
    noise_points = np.sum(clusters == -1)

    traits['Total Clusters Detected'] = num_clusters
    traits['Noise Points'] = noise_points

    # 6. CALCULATE LEAF ANGLES (2.5D approach)
    df_leaf_angles = calculate_leaf_angles_2_5d(points, clusters, stem_center_xy)

    # Filter by minimum points
    df_leaf_angles_filtered = df_leaf_angles[df_leaf_angles['Num_Points'] >= min_leaf_points]

    if len(df_leaf_angles_filtered) > 0:
        traits['Average Distance from Center'] = df_leaf_angles_filtered['Distance_from_Center'].mean()
        traits['Max Distance from Center'] = df_leaf_angles_filtered['Distance_from_Center'].max()
        traits['Average Tilt Angle'] = df_leaf_angles_filtered['Tilt_from_Horizontal'].mean()
        traits['Average Leaf Length'] = df_leaf_angles_filtered['Leaf_Length'].mean()
        traits['Average Leaf Width'] = df_leaf_angles_filtered['Leaf_Width'].mean()

    # 7. ANALYZE EACH CLUSTER
    cluster_info = []
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue

        cluster_points = points[clusters == cluster_id]
        cluster_size = len(cluster_points)

        cluster_width_x = cluster_points[:, 0].max() - cluster_points[:, 0].min()
        cluster_width_y = cluster_points[:, 1].max() - cluster_points[:, 1].min()
        cluster_area = cluster_width_x * cluster_width_y
        cluster_depth = cluster_points[:, 2].max() - cluster_points[:, 2].min()

        center = cluster_points.mean(axis=0)
        distance_from_stem = np.linalg.norm(center[:2] - stem_center_xy)

        # Get angle if available
        angle_info = df_leaf_angles[df_leaf_angles['Cluster_ID'] == cluster_id]
        azimuth = angle_info['Azimuthal_Angle'].values[0] if len(angle_info) > 0 else None
        tilt = angle_info['Tilt_from_Horizontal'].values[0] if len(angle_info) > 0 else None

        cluster_info.append({
            'Cluster ID': cluster_id,
            'Points': cluster_size,
            'Azimuthal Angle (°)': round(azimuth, 2) if azimuth is not None else 'N/A',
            'Tilt Angle (°)': round(tilt, 2) if tilt is not None else 'N/A',
            'Distance from Center': round(distance_from_stem, 2),
            'Width (X)': round(cluster_width_x, 2),
            'Width (Y)': round(cluster_width_y, 2),
            'Area Estimate': round(cluster_area, 2),
            'Depth Variation (Z)': round(cluster_depth, 2)
        })

    df_clusters = pd.DataFrame(cluster_info)
    df_clusters = df_clusters.sort_values('Points', ascending=False)

    # 8. NUMBER OF LEAVES
    potential_leaves = df_clusters[df_clusters['Points'] >= min_leaf_points]
    num_leaves = len(potential_leaves)
    traits['Estimated Number of Leaves'] = num_leaves

    # 9. ADDITIONAL TRAITS
    traits['Total Points'] = len(points)
    traits['Compactness'] = canopy_area_bbox / (np.pi * (canopy_diameter/2)**2) if canopy_diameter > 0 else 0

    # Radial distribution (distance from center)
    distances_from_center = np.linalg.norm(points[:, :2] - stem_center_xy, axis=1)
    max_radius = distances_from_center.max()

    inner_ring = np.sum(distances_from_center < max_radius/3)
    middle_ring = np.sum((distances_from_center >= max_radius/3) & (distances_from_center < 2*max_radius/3))
    outer_ring = np.sum(distances_from_center >= 2*max_radius/3)

    traits['Inner Ring (%)'] = round(100 * inner_ring / len(points), 2)
    traits['Middle Ring (%)'] = round(100 * middle_ring / len(points), 2)
    traits['Outer Ring (%)'] = round(100 * outer_ring / len(points), 2)

    return traits, df_clusters, clusters, stem_center_xy, df_leaf_angles_filtered

# Sidebar
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload PLY File", type=['ply'])

st.sidebar.markdown("### Clustering Parameters")
eps = st.sidebar.slider("DBSCAN eps (leaf separation)", 0.01, 0.20, 0.08, 0.01)
min_samples = st.sidebar.slider("Min samples per cluster", 5, 30, 10, 1)
min_leaf_points = st.sidebar.slider("Min points to count as leaf", 10, 50, 20, 5)

# Main content
if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_pointcloud.ply", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load point cloud
    with st.spinner("Loading point cloud..."):
        try:
            points_data, properties = read_ply_file("temp_pointcloud.ply")
            points = points_data[:, :3]

            st.success(f"✅ Successfully loaded {len(points)} points!")

            # Display basic info with corrected interpretation
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Points", len(points))
            with col2:
                x_range = points[:, 0].max() - points[:, 0].min()
                st.metric("X Range (Width)", f"{x_range:.1f}")
            with col3:
                y_range = points[:, 1].max() - points[:, 1].min()
                st.metric("Y Range (Depth)", f"{y_range:.1f}")
            with col4:
                z_range = points[:, 2].max() - points[:, 2].min()
                st.metric("Z Range (Depth var)", f"{z_range:.1f}")

            st.info("📸 This is a 2.5D top-down view: X-Y plane shows plant spread, Z shows depth from camera")

        except Exception as e:
            st.error(f"Error loading PLY file: {e}")
            st.stop()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 3D View", "🗺️ Top-Down View", "🔬 Plant Traits", "📐 Leaf Analysis", "📋 Cluster Details"])

    with tab1:
        st.markdown("## 3D Point Cloud Visualization")

        with st.spinner("Rendering 3D point cloud..."):
            fig = create_3d_pointcloud_plot(points, title=f"3D Point Cloud - {uploaded_file.name}")
            st.plotly_chart(fig, use_container_width=True)

        st.info("💡 X-Y plane = horizontal plant spread, Z = depth variation from camera")

    with tab2:
        st.markdown("## Top-Down View (Bird's Eye)")

        if st.button("🔬 Run Complete Analysis", type="primary", key="analyze_btn"):
            with st.spinner("Analyzing plant traits..."):
                (traits, df_clusters, clusters, stem_center, 
                 df_leaf_angles) = analyze_plant_phenotype_2_5d(
                    points, eps=eps, min_samples=min_samples, min_leaf_points=min_leaf_points
                )

                st.session_state['traits'] = traits
                st.session_state['df_clusters'] = df_clusters
                st.session_state['clusters'] = clusters
                st.session_state['stem_center'] = stem_center
                st.session_state['df_leaf_angles'] = df_leaf_angles

        if 'stem_center' in st.session_state:
            fig_topdown = create_top_down_view(
                points,
                clusters=st.session_state.get('clusters'),
                stem_center=st.session_state['stem_center'],
                leaf_angles_df=st.session_state.get('df_leaf_angles')
            )
            st.plotly_chart(fig_topdown, use_container_width=True)
            st.info("🎯 Red star = stem center, Orange lines = leaf orientations from center")
        else:
            st.info("👆 Run analysis to see top-down view with leaf orientations")

    with tab3:
        st.markdown("## Plant Phenotyping Analysis (2.5D)")

        if 'traits' in st.session_state:
            traits = st.session_state['traits']

            st.markdown("### 📏 Primary Traits")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Canopy Diameter", f"{traits['Canopy Diameter']:.2f}", help="Diagonal measurement in XY plane")
            with col2:
                st.metric("Number of Leaves", traits['Estimated Number of Leaves'])
            with col3:
                avg_dist = traits.get('Average Distance from Center', 0)
                st.metric("Avg Leaf Distance", f"{avg_dist:.1f}", help="From stem center")
            with col4:
                area = traits['Canopy Area (BBox)']
                st.metric("Canopy Area", f"{area:.0f}", help="Bounding box area")

            st.markdown("### 📊 Complete Traits Table")

            traits_df = pd.DataFrame({
                'Trait': list(traits.keys()),
                'Value': [f"{v:.4f}" if isinstance(v, float) else str(v) for v in traits.values()]
            })

            st.dataframe(traits_df, use_container_width=True, height=500)

            csv = traits_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Traits CSV",
                data=csv,
                file_name="plant_traits_2_5d.csv",
                mime="text/csv"
            )

        else:
            st.info("👆 Run the analysis from the Top-Down View tab first")

    with tab4:
        st.markdown("## 📐 Leaf Orientation Analysis (2.5D)")

        if 'df_leaf_angles' in st.session_state:
            df_leaf_angles = st.session_state['df_leaf_angles']

            if len(df_leaf_angles) > 0:
                st.markdown(f"### Detected {len(df_leaf_angles)} leaves")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Avg Distance", f"{df_leaf_angles['Distance_from_Center'].mean():.1f}")
                with col2:
                    st.metric("Max Distance", f"{df_leaf_angles['Distance_from_Center'].max():.1f}")
                with col3:
                    st.metric("Avg Tilt", f"{df_leaf_angles['Tilt_from_Horizontal'].mean():.1f}°")
                with col4:
                    st.metric("Avg Leaf Length", f"{df_leaf_angles['Leaf_Length'].mean():.1f}")

                st.markdown("### 📊 Leaf Details")
                st.info("""
                **Azimuthal Angle:** Direction of leaf from stem center in XY plane (0-360°)  
                **Tilt Angle:** How much leaf surface tilts from horizontal (0° = flat, ±90° = vertical)
                """)

                df_display = df_leaf_angles.sort_values('Cluster_ID')
                st.dataframe(df_display, use_container_width=True, height=400)

                csv = df_leaf_angles.to_csv(index=False)
                st.download_button(
                    label="📥 Download Leaf Angles CSV",
                    data=csv,
                    file_name="leaf_angles_2_5d.csv",
                    mime="text/csv"
                )

                # Polar plot of leaf positions
                st.markdown("### 🎯 Leaf Arrangement (Polar View)")

                import plotly.express as px
                fig_polar = go.Figure()

                fig_polar.add_trace(go.Scatterpolar(
                    r=df_leaf_angles['Distance_from_Center'],
                    theta=df_leaf_angles['Azimuthal_Angle'],
                    mode='markers+text',
                    text=df_leaf_angles['Cluster_ID'],
                    textposition='top center',
                    marker=dict(size=15, color=df_leaf_angles['Tilt_from_Horizontal'], 
                               colorscale='Viridis', showscale=True,
                               colorbar=dict(title="Tilt Angle")),
                    name='Leaves'
                ))

                fig_polar.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title="Leaf Positions Around Stem Center"
                )

                st.plotly_chart(fig_polar, use_container_width=True)

            else:
                st.warning("No leaves detected with sufficient points")
        else:
            st.info("Run the analysis first")

    with tab5:
        st.markdown("## Cluster Details")

        if 'df_clusters' in st.session_state:
            df_clusters = st.session_state['df_clusters']

            st.markdown(f"### Found {len(df_clusters)} clusters")
            st.dataframe(df_clusters, use_container_width=True, height=400)

            csv = df_clusters.to_csv(index=False)
            st.download_button(
                label="📥 Download Cluster Details CSV",
                data=csv,
                file_name="cluster_details_2_5d.csv",
                mime="text/csv"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Largest Cluster", f"{df_clusters['Points'].max()} points")
                st.metric("Average Cluster Size", f"{df_clusters['Points'].mean():.1f} points")

            with col2:
                st.metric("Smallest Cluster", f"{df_clusters['Points'].min()} points")
                st.metric("Total Clustered Points", df_clusters['Points'].sum())
        else:
            st.info("Run the analysis first")

else:
    st.info("👈 Please upload a PLY file using the sidebar to begin analysis")

    st.markdown("""
    ### 2.5D Top-Down Plant Phenotyping

    **Optimized for RGB-D/Depth cameras viewing plants from above**

    ### How to Use:
    1. **Upload** your PLY point cloud file (top-down view)
    2. **View** 3D and top-down visualizations
    3. **Run Analysis** to extract traits and leaf orientations
    4. **Download** CSV reports

    ### Features:
    - 🌿 3D and 2D top-down visualizations
    - 📏 Canopy area, diameter, and spread measurements
    - 🍃 Leaf counting and segmentation
    - 📐 **Leaf azimuthal angles** (direction from stem)
    - 📐 **Leaf tilt angles** (surface orientation)
    - 🎯 Stem center identification
    - 🗺️ Polar plot showing leaf arrangement

    ### Measurements (2.5D Specific):
    - **Canopy Diameter & Area** (primary size metric)
    - **Leaf Azimuthal Angles** (0-360° from stem center)
    - **Leaf Tilt Angles** (tilt from horizontal plane)
    - **Distance from Center** (radial spread)
    - **Radial Distribution** (inner/middle/outer rings)

    ### Understanding 2.5D:
    - **X-Y Plane:** Where the plant spreads horizontally
    - **Z Axis:** Small depth variations from camera
    - **Stem:** Located at center of XY plane
    - **Leaves:** Radiate outward in XY plane
    """)

