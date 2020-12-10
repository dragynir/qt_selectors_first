import numpy as np
import pandas as pd
from scipy import spatial
from scipy.spatial import distance

import bokeh.layouts
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Button, TextInput, ColorBar
from bokeh.models import Panel, Tabs, Select, CustomJS, ContinuousTicker, RadioButtonGroup, CheckboxGroup, RangeSlider
from bokeh.plotting import figure, output_notebook, output_file, show
from bokeh.themes import Theme
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature
from functools import partial
import bokeh.transform as tr
import bokeh.palettes
from bokeh.palettes import Greys256, Turbo256, Magma256, Cividis256, Viridis256, Plasma256, Inferno256, Blues8
from bokeh.palettes import Category10, Category20, Set1, Set2, Paired12
from bokeh.models.tools import HoverTool, TapTool

import datashader as ds
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize


from bokeh.io import curdoc
from os.path import dirname, join
import os


from bokeh.io import export_png

hv.extension('matplotlib')



__all__ = ['plot_maps', 'cluster_selector', 'hist_cluster_selector', 'create_cluster_pipeline']


def plot_maps(df, kdims=['X', 'Y'], vdims=['Elev'], ncols=1, fig_size=125, 
              raster_width=150, raster_height=150, raster_aggr=ds.mean, 
              x_sampl=1, y_sampl=1, colormap='viridis', backend='matplotlib', 
              height=600, width=900, dynamic=False, **kwargs):
    maps = hv.Points(df, kdims=kdims, vdims=vdims)
    
    shaded = [rasterize(maps, 
                        width=raster_width, 
                        height=raster_height,
                        aggregator=raster_aggr(k),
                        dynamic=dynamic,
                        x_sampling=x_sampl,
                        y_sampling=y_sampl, 
                        precompute=True,
                      ).opts(title=k, colorbar=True, cmap=colormap) for k in vdims]

    SH = shaded[0]
    for sh in shaded[1:]:
        SH += sh
        
    if backend == 'matplotlib':    
        kwargs = dict(fig_size=fig_size)
    else:
        kwargs = dict(height=height, width=width)
    
    if ncols == 1:
        return SH.opts(**kwargs)
    else:
        return SH.cols(ncols).opts(**kwargs)


class _ClustersSelector():
    
    def __init__(self, doc, df, col, traces, traces_data, xc, yc, x, y, cmap, map_cmaps, max_selections, map_size, mode, callback, save_name):
        
        self.df = df
        self.save_name = save_name
        self.col = col
        self.traces = traces
        self.traces_data = traces_data
        
        
        self.show_traces_figure = True
        if self.traces is None or self.traces_data is None:
            self.show_traces_figure = False
        
        
        self.attr = self.col[0]
        self.df['clusters'] = np.zeros(df.shape[0])
        
        self.source = ColumnDataSource(data=self.df)
        self.source.data['colors'] = np.zeros(df.shape[0])
        self.source.data['attr_values'] = df[self.col[0]]
        
        self.map_tools = "pan,wheel_zoom,box_zoom,lasso_select,box_select,reset,save"
        self.colors = [cmap[i] for i in range(0, len(cmap), len(cmap) // (max_selections))]
        self.map_cmaps = map_cmaps
        self.cmap_interactive = tr.linear_cmap('colors', self.colors, low=0 , high=len(self.colors))
        self.cmap_interactive_map = tr.linear_cmap('clusters', self.colors, low=0 , high=len(self.colors))
        
        btn_w = map_size[0] // 3
        self.confirm_btn = Button(label="Confirm", button_type="success", max_width=btn_w)
        self.cancel_btn = Button(label="Cancel", button_type="success", max_width=btn_w)
        self.new_btn = Button(label="New cluster", button_type="success", max_width=btn_w)
        
        self.attr_select = Select(title="Attribute", options=col)
        
        self.curr_cluster = 1
        self.clusters_count = 1
        self.clusters_dict = dict()

        self.confirm_btn.on_click(self.confirm_cluster)
        self.cancel_btn.on_click(self.cancel_cluster)
        self.new_btn.on_click(self.new_cluster)
        
        self.clusters = figure(plot_width=map_size[0], plot_height=map_size[1], tools=self.map_tools, output_backend='webgl')
        self.clusters.circle(xc, yc, source=self.source, color=self.cmap_interactive, alpha=0.8, size=1.5)
        self.clusters.title.text = "UMAP embedding"
        
        self.cluster_map = figure(plot_width=map_size[0], plot_height=map_size[1], tools=self.map_tools, output_backend='webgl')
        self.cluster_map.square(x, y, source=self.source, color=self.cmap_interactive_map, size=3)
        self.cluster_map.title.text = " Current cluster color "
        self.cluster_map.title.background_fill_color = self.colors[1]
        
        self.selected_s = None
        
        self.source.selected.on_change('indices', self.indices_selected)
        
        
        self.create_attr_map(x, y, map_size)        
        self.cluster_mode = 1
        self.clusters_mode_group = RadioButtonGroup(labels=['Attribute mode', 'Cluster mode'], 
                                                    active=1, max_width=map_size[0])
        
    
        self.clusters_mode_group.on_change('active', self.change_mode) 
        
        self.clusters_group = RadioButtonGroup(labels=['Cluster 1'], active=0, max_width=map_size[0])
        
        self.clusters_group.on_change('active', self.change_cluster)
        
        control_layout = column(row(self.cancel_btn, self.confirm_btn, self.new_btn), 
                                self.clusters_group, self.clusters_mode_group)
        attr_map_layout = column(self.select_cmap, self.select_attr)
        
        if self.show_traces_figure:     
            self.traces_figure = self.create_traces_plot(map_size)

            self.cluster_map.add_tools(TapTool(renderers=[self.source_dots]))
            self.source_checkbox = CheckboxGroup(labels=["Source", "Group"])
            self.source_checkbox.on_change('active', self.source_checkbox_callback)

            self.plot_traces(self.source_traces_dict[0])
            self.selected_source = self.source_traces_dict[0]
            
            self.traces_layout = row(column(self.gain_slider, self.cliping_slider), self.traces_figure)
        else:
            self.source_checkbox = None
            self.traces_layout = None
        
        self.cluster_layout = gridplot([[control_layout, self.source_checkbox, attr_map_layout], 
                                [self.clusters, self.cluster_map, self.static_map_figure]])


        
        if self.traces_layout is None:
            self.layout = self.cluster_layout
        else:
            self.layout = column(self.cluster_layout, self.traces_layout) 

        if mode == 'server':
            self.layout = column(self.__create_save_panel(), self.layout)
        elif mode == 'embed':
            self.layout = column(self.__create_file_save_panel(), self.layout) 

        if not mode == 'embed':           
            doc.add_root(self.layout)
        # else use self.layout


    def __create_file_save_panel(self):
        save_btn = Button(label="Save result dataframe", button_type="success", width_policy='min')
        save_btn.on_click(self.__save_dataframe)
        return save_btn

    def __create_save_panel(self):
        save_btn = Button(label="Save result dataframe", button_type="success", width_policy='min')
        save_btn.on_click(self.__save_dataframe)

        download_btn = Button(label="Download result dataframe", button_type="success", width_policy='min')
        download_btn.js_on_click(CustomJS(args=dict(source=ColumnDataSource(self.df)),
                                    code=open(join(dirname(__file__), 'download_df.js')).read()))
        return row(save_btn, download_btn)

    def __save_dataframe(self):
        self.df.to_csv(self.save_name)
    
    
    def clip_gain_slider_callback(self, attr, old, new):
        self.plot_traces(self.selected_source)
        
    
    def source_tapselected_callback(self, attr, old, new):
        if len(new) == 0:
            return
        self.selected_source = self.source_traces_dict[new[0]]
        self.plot_traces(self.selected_source)
        
       
    def source_checkbox_callback(self, attr, old, new):    
        if 0 in new:     
            self.source_dots.visible = True
        else:
            self.source_dots.visible = False
            self.curr_group.visible = False
        if 1 not in new:
            self.curr_group.visible = False
            
    def create_traces_plot(self, map_size):
        v = np.column_stack((self.traces['SourceX'].values, self.traces['SourceY'].values))
        self.unique_source = np.unique(v, axis=0)
        
        self.source_traces_dict = dict()
        
        for i, src_coords in enumerate(self.unique_source):
            tr_data, y, vis_traces  = self.find_source_traces(src_coords[0], src_coords[1])
            self.source_traces_dict[i] = {'data':tr_data, 'y': y, 'traces': vis_traces}
        
        
        self.traces_source_columnsource = ColumnDataSource(data={'x': self.unique_source[:,0], 'y': self.unique_source[:,1]})

        fig_tools = "pan,wheel_zoom,box_zoom,reset,save"
        self.traces_figure = figure(plot_width=map_size[0] * 2, plot_height=map_size[1] * 2, tools=fig_tools, output_backend='webgl')
        self.curr_traces = None
        self.curr_group = self.cluster_map.square(x=[], y=[], color='blue', size=3)
        self.source_dots = self.cluster_map.circle(x='x', y='y', source=self.traces_source_columnsource, color='green', size=10)
        self.source_dots.visible = False
        
    
        self.gain_slider = Slider(start=0, end=1, value=0, step=0.1, title='Gain: ')
        self.gain_slider.on_change('value_throttled', self.clip_gain_slider_callback)
        
        self.cliping_slider = RangeSlider(start=-10, end=10, value=(-9,9), step=.5, title="Clip: ")
        self.cliping_slider.on_change('value_throttled', self.clip_gain_slider_callback)
    
        
        self.traces_source_columnsource.selected.on_change('indices', self.source_tapselected_callback)    
        
        return self.traces_figure
                                       
                       
    def find_source_traces(self, src_x, src_y):
        
        traces = self.traces
                                       
        source_coords = np.column_stack((traces['SourceX'].values, traces['SourceY'].values))
        
        pt = [(src_x, src_y)]
        source_coords = source_coords.astype(np.float32)
        
        i = np.argmin(distance.cdist(pt, source_coords, 'euclidean').squeeze())
        
        
        src_pos = source_coords[i]
        source_id = traces['EnergySourcePoint'].values[i]
        
        traces = traces[(traces['EnergySourcePoint'] == source_id)]        
        
        g_source = np.column_stack((traces['GroupX'].values, traces['GroupY'].values))
        
        g_source = g_source.astype(np.float32)
    
        i = np.argmin(distance.cdist([src_pos], g_source, 'euclidean').squeeze())
    
        
        xline = traces['XLINE_NO'].values[i]
        
    
        vis_traces = traces[(traces['XLINE_NO'] == xline)]
        
        show_data = self.traces_data[vis_traces['TRACE_SEQUENCE_LINE'].values]
 
        tr_data = show_data.copy()
        y = np.arange(tr_data.shape[1])[None, ...]
        y = np.repeat(y, tr_data.shape[0], axis=0)   
                                       
        return tr_data, y, vis_traces  
    
            
    def plot_traces(self, source_dict):
        
        tr_data, y, vis_traces = source_dict['data'], source_dict['y'], source_dict['traces']
        
        
        tr_data = tr_data * (1 + self.gain_slider.value)
        tr_data = np.clip(tr_data, self.cliping_slider.value[0], self.cliping_slider.value[1])
        
        for i in range(len(tr_data)):
            tr_data[i] = tr_data[i][::-1] + (i * 10)
                
        if self.curr_traces is not None:
            self.curr_traces.visible = False
        
        
        if 1 in self.source_checkbox.active:
            self.curr_group.visible = False
            self.curr_group = self.cluster_map.square(x=vis_traces['GroupX'], y=vis_traces['GroupY'], color='blue', size=3)
        
        self.curr_traces = self.traces_figure.multi_line(xs=list(tr_data), ys=list(y), color='black', line_width=1)
        
        
    def change_mode(self, attr, old, new):
        self.source.selected.indices = []
        self.cluster_mode = new
        if new == 0:
            self.cmap_interactive['transform'].palette = self.map_cmaps[self.select_cmap.value]
            self.cmap_interactive['transform'].high = None
            self.cmap_interactive['transform'].low = None
            self.source.data['colors'] = self.source.data['attr_values']
        elif new == 1:
            self.cmap_interactive['transform'].palette = self.colors
            self.cmap_interactive['transform'].high = len(self.colors)
            self.cmap_interactive['transform'].low = 0
            self.source.data['colors'] = self.source.data['clusters']
        else: pass
        
    def confirm_cluster(self):
        if (self.curr_cluster in self.clusters_dict.keys()):
            new_clusters = self.source.data['clusters']
            new_clusters[self.clusters_dict[self.curr_cluster]] = self.curr_cluster
            self.source.data['clusters'] = new_clusters
            self.df['clusters'] = new_clusters
            if self.cluster_mode == 1:
                self.source.data['colors'] = new_clusters
            self.source.selected.indices = []
            
    def cancel_cluster(self):
        if (self.curr_cluster in self.clusters_dict.keys()):
            new_clusters = self.source.data['clusters']
            new_clusters[new_clusters == self.curr_cluster] = 0
            self.source.data['clusters'] = new_clusters
            self.df['clusters'] = new_clusters
            if self.cluster_mode == 1:
                self.source.data['colors'] = new_clusters
             
    def change_cluster(self, attr, old, new):
        self.curr_cluster = new + 1
        self.cluster_map.title.background_fill_color = self.colors[self.curr_cluster]
        self.source.selected.indices = []
        
    def create_attr_map(self, x, y, map_size):
        colors_attr = list(self.map_cmaps.keys())
        self.cmap_static = tr.linear_cmap('attr_values', self.map_cmaps[colors_attr[0]], None, None)
        
        self.static_map_figure = figure(title='Map', tools=self.map_tools,
                    plot_width=map_size[0], plot_height=map_size[1], output_backend='webgl')
        self.static_map_figure.square(x=x, y=y, source=self.source, color=self.cmap_static, size=3)
        
        color_bar = ColorBar(color_mapper=self.cmap_static['transform'], width=10, location=(0,0))
        self.static_map_figure.add_layout(color_bar, 'right')
        
        self.select_cmap = Select(title="Colormap:", value=colors_attr[0], options=colors_attr)
        self.select_attr = Select(title="Attribute:", value=self.col[0], options=self.col)
        
        def select_color_callback(call_attr, old, new, cs):
            cs.select_color(call_attr, old, new) 
            
        def select_attr_callback(call_attr, old, new, cs):
            cs.select_attribute(call_attr, old, new)
        
        self.select_cmap.on_change('value', partial(select_color_callback, cs=self))   
        self.select_attr.on_change('value', partial(select_attr_callback, cs=self)) 
        
    def select_color(self, call_attr, old, new):
        self.cmap_static['transform'].palette = self.map_cmaps[new]
        if self.cluster_mode == 0:
            self.cmap_interactive['transform'].palette = self.map_cmaps[new]
        
    def select_attribute(self, call_attr, old, new):
        self.source.data['attr_values'] = self.df[new]
        if self.cluster_mode == 0: 
            self.source.data['colors'] = self.df[new]
           
    def indices_selected(self, attr, old, new):
        self.clusters_dict[self.curr_cluster] = new
         
    def new_cluster(self):
        if (self.curr_cluster in self.clusters_dict.keys()) and self.clusters_count <= len(self.colors):
            self.clusters_count += 1
            self.curr_cluster = self.clusters_count
            self.clusters_group.labels.append('Cluster ' + str(self.clusters_count))
            self.cluster_map.title.background_fill_color = self.colors[self.clusters_count]
            self.clusters_group.active = self.clusters_count - 1
       
    
def cluster_selector(df, cols, headers=None, traces=None, k_map=None, k_emb=None, map_cmaps=None, 
                            max_selections=7, map_size=(400, 400), mode='notebook', save_name='result_df.csv', callback=None,):
    """
    Creates interactive tool for manual clusterization by lasso and box selection with saving into the source dataframe
   
    Parameters
    ----------
    Input:
        df - pandas.DataFrame; Table of data with attributes
        
        cols - list of str; Names of columns (attributes) in "df" to be plotted
    
        headers - DataFrame; Trace headers for seismograms that are plotted as DataFrame with fields
                    SourceX, SourceY, EnergySourcePoint, GroupX, GroupY, XLINE_NO, TRACE_SEQUENCE_LINE

        traces - 2D numpy.ndarray; Traces (gathers) to be plotted
            
        k_map - list of two strings; For example ['X', 'Y']. Two keywords that indicate column names with coordinates
            
        k_emb - list of two strings; For example ['F1', 'F2']. Two keywords that indicate column names with coordinates of UMAP embedding
            
        map_cmaps - dict {cmap_name: cmap} or None; Dictionary with colormaps to be used for visualizing maps of attributes
                                                    First cmap in the dict is for clusters colors 
            
        max_selections - int; Max number of cluster selections, e.g. 7 is set by default.
            
        map_size - tuple of two ints; Means size of each map in the figure, e.g. (400, 400) is set by default.

        mode - 'notebook', 'server', 'embed'; Use mode 'notebook' in jupyter notebook like environment.
                                             Use mode 'server' to launch bokeh app (bokeh serve --show script.py)
                                             Mode 'embed' is only for ClusterPipeline.

        save_name - path to save source dataframe, e.g. 'result_df.csv'is set by default

    Output:
        in mode 'notebook': fig: bokeh.models.layout. This is plotted via bokeh.plotting.show
        in mode 'server': None
        in mode 'embed': returns _ClustersSelector() object. Use _ClustersSelector().layout to 
        get bokeh layout and embed tool to another bokeh layout	
    """

    
    if map_cmaps is None:
        map_cmaps = {'Turbo': Turbo256, 'Viridis': Viridis256, 'Magma': Magma256, 
                     'Plasma': Plasma256, 'Cividis': Cividis256, 'Inferno': Inferno256, 'Greys': Greys256}

    cmap = list(map_cmaps.values())[0]
    
    if k_map is None:
        k_map = ['X', 'Y']
        
    if k_emb is None:
        k_emb = ['X', 'Y']

    if mode == 'notebook':
        output_notebook()
        def launch(doc):       
            cs = _ClustersSelector(doc=doc, df=df, traces=headers, traces_data=traces, col=cols, xc=k_emb[0], yc=k_emb[1], 
                                  x=k_map[0], y=k_map[1], cmap=cmap, map_cmaps=map_cmaps, 
                                  max_selections=max_selections, map_size=map_size, mode=mode, callback=callback, save_name=save_name)
        return launch

    elif mode == 'server':
        output_file('out.html')
        doc = curdoc()
        cs = _ClustersSelector(doc=doc, df=df, traces=headers, traces_data=traces, col=cols, xc=k_emb[0], yc=k_emb[1], 
                                  x=k_map[0], y=k_map[1], cmap=cmap, map_cmaps=map_cmaps, 
                                  max_selections=max_selections, map_size=map_size, mode=mode, callback=callback, save_name=save_name)
    elif mode == 'embed':
        cs = _ClustersSelector(doc=None, df=df, traces=headers, traces_data=traces, col=cols, xc=k_emb[0], yc=k_emb[1], 
                                  x=k_map[0], y=k_map[1], cmap=cmap, map_cmaps=map_cmaps, 
                                  max_selections=max_selections, map_size=map_size, mode=mode, callback=callback, save_name=save_name)
        return cs
    else:
        print('Unknown launch mode. Available modes are notebook, server, embed')

    return None




class _HistClusterSelector():
    
    
    def __init__(self, doc, df, attributes, map_cmaps, max_sliders_count, hist_cmap,  hist_size, map_size, mode, callback, save_name):
        self.histo_bins_count = 60
        self.sliders_count = 1
        self.max_sliders_count = max_sliders_count
        self.attributes = attributes
        self.df = df
        self.save_name = save_name
        self.map_cmaps = map_cmaps
        self.sliders_split_bins = []
        self.attr_tab_dict = dict()
        self.tabs_list = []
        self.confirmed = True
        
           
        # create palette
        self.colors = [hist_cmap[i] for i in range(0, len(hist_cmap), len(hist_cmap)//(self.max_sliders_count + 1))]

        self.hists_dict = dict()
        
        for key in attributes:
            hist, edges = np.histogram(df[key].values, self.histo_bins_count)
            self.hists_dict[key] = [hist, edges]
        
        
        self.source_histo, self.hist_figure = self.create_hist(hist_size)
        
        
        self.source_map = ColumnDataSource(data=self.df)
        self.source_map.data['fill_map'] = np.zeros(self.df.shape[0])

        # values for clusters map
        self.source_map.data['map_colors'] = np.zeros(self.df.shape[0])

        # values for attribute map
        self.source_map.data['map_values'] = self.df[self.attributes[0]]
        
        self.cmap_interactive, self.cluster_map_figure = self.create_clusters_map(map_size)
        
        self.attr_map_figure, self.cmap_attr_map, self.attr_colors = self.create_attr_map(map_size)


        
        self.control_panel = self.create_control_panel(mode, callback)
        
            
        self.layout = gridplot([[self.selection_group, None, None],
            [self.control_panel, self.hist_figure, self.cluster_map_figure, self.attr_map_figure]])


        if not mode == 'embed':
            doc.add_root(self.layout)

        # else use self.layout


    def __create_save_panel(self):
        save_btn = Button(label="Save result dataframe", button_type="success", width_policy='min')
        save_btn.on_click(self.__save_dataframe)

        download_btn = Button(label="Download result dataframe", button_type="success", width_policy='min')
        download_btn.js_on_click(CustomJS(args=dict(source=ColumnDataSource(self.df)),
                                    code=open(join(dirname(__file__), 'download_df.js')).read()))
        return row(save_btn, download_btn)

    def __save_dataframe(self):
        self.df.to_csv(self.save_name)
        
        
    def create_hist(self, hist_size):
        hist_tools = "pan,wheel_zoom,box_zoom,box_select,reset"
        hist_figure = figure(title='Histogram', tools=hist_tools,
                     plot_width=hist_size[0],
                     plot_height=hist_size[1], tooltips=[("count", "(@hist)")])
        
        
        
        start_hist, start_edges = self.hists_dict[self.attributes[0]]
        source_histo = ColumnDataSource(data={
                                        'hist': start_hist,
                                        'left': start_edges[:-1],
                                        'right': start_edges[1:],   
                                        'color': np.full(len(start_edges) - 1, self.colors[0])
                                       })
        hist_figure.quad(top='hist', bottom=0, left='left', right='right',
                         source=source_histo, fill_color='color', line_color="white", alpha=0.7)
        
        return source_histo, hist_figure
    
    def create_clusters_map(self, map_size):
        
        cmap_interactive = tr.linear_cmap('map_colors', self.colors, low=0 , high=len(self.colors))
                
        map_tools = "pan,wheel_zoom,box_zoom,lasso_select,box_select,reset,save"
        hover_tooltips=[
             ("(x,y)", "(@X, @Y)"),
            ("value", "@map_values"),
        ]
        
        cluster_map_figure = figure(title='Map', tools=map_tools,
                            plot_width=map_size[0], plot_height=map_size[1], output_backend='webgl', tooltips=hover_tooltips)
        
        cluster_map_figure.circle(x='X', y='Y', source=self.source_map, color=cmap_interactive, alpha=0.7)
        
        return cmap_interactive, cluster_map_figure
    
    
    def create_attr_map(self, map_size):
        
        attr_colors = list(self.map_cmaps.keys())
        cmap_attr_map = tr.linear_cmap('map_values', self.map_cmaps[attr_colors[0]], None, None)
        
        map_tools = "pan,wheel_zoom,box_zoom,lasso_select,box_select,reset,save"
        hover_tooltips=[
             ("(x,y)", "(@X, @Y)"),
            ("value", "@map_values"),
        ]
            
            
        attr_map_figure = figure(title='Map', tools=map_tools,
                    plot_width=map_size[0], plot_height=map_size[1], output_backend='webgl', tooltips=hover_tooltips)
        
        
        attr_map_figure.circle(x='X', y='Y', source=self.source_map, color=cmap_attr_map, alpha=0.7)
        
        color_bar = ColorBar(color_mapper=cmap_attr_map['transform'], width=8,  location=(0,0))
            
        attr_map_figure.add_layout(color_bar, 'left')
        
        return attr_map_figure, cmap_attr_map, attr_colors
 
    
    def select_attr_cmap_callback(self, call_attr, old, new):
         self.cmap_attr_map['transform'].palette = self.map_cmaps[new]

    
    def get_bins_split(self, min_edge):
        # return split values from histogram
        bins = [min_edge]
        prev_value = bins[0];
        for sl in self.sliders:
            if sl.value > prev_value:
                bins.append(sl.value)
                prev_value = sl.value  
        return bins
    
    
    def sliders_hist_callback(self, attr, old, new):
        min_edge = self.source_histo.data['left'][0]
        max_edge = self.source_histo.data['right'][-1]
    
        bins = self.get_bins_split(min_edge)
                
        self.sliders_split_bins = bins[1:]
                
        hist_split = [0]
        max_cols = len(self.source_histo.data['color'])
        for b in bins[1:]:
            # calculate hist column according to slider value
            index = int(round((max_cols - 1) * abs((b - min_edge) / (max_edge - min_edge))))
            hist_split.append(index)
            

        new_colors = self.source_histo.data['color']
        prev_v = 0
        for i in range(len(hist_split)):
            new_colors[prev_v : hist_split[i]] = self.colors[i]
            prev_v = hist_split[i]
            
        new_colors[hist_split[-1]:max_cols] = self.colors[0]
        
        self.source_histo.data['color'] = new_colors
    
        
    def sliders_map_callback(self, attr, old, new):
        
        min_edge = self.source_histo.data['left'][0]
        
        bins = self.get_bins_split(min_edge)
            
        attr_name = self.select_attr.value
        
        map_data = np.zeros(len(self.source_map.data['map_colors']))

        for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
            value_mask = (self.df[attr_name] > left) & (self.df[attr_name] < right)
            map_data[value_mask] = i + 1
            
        self.source_map.data['map_colors'] = map_data
       
        
        
    def reset_sliders(self, step, edges):
        for sl in self.sliders:
            sl.start = edges[0]
            sl.end = edges[-1]
            sl.value = edges[0]
            sl.step = step

            
    
    def bins_input_callback(self, attr, old, new):
        if not new.isdigit():
            return
        self.histo_bins_count = int(new)
        
        attr_name = self.select_attr.value
        hist, edges = np.histogram(self.df[attr_name].values, self.histo_bins_count)    
        self.hists_dict[attr_name] = [hist, edges]
        
        new_data = {'hist': hist, 'left':edges[:-1], 'right':edges[1:], 'color': np.full(len(edges) - 1, self.colors[0])}    
        self.source_histo.data = new_data
        
        step = abs(edges[1] - edges[0])
        self.source_map.data['map_colors'] = self.source_map.data['fill_map']
        self.reset_sliders(step, edges)
       
    

    
    def select_attr_callback(self, call_attr, old, new):
        attr_name = new
        hist, edges = self.hists_dict[attr_name]
        
        new_data = data={
                        'hist': hist,
                        'left': edges[:-1],
                        'right': edges[1:],   
                        'color': np.full(len(edges) - 1, self.colors[0])
        
                       }
        self.source_histo.data = new_data

        self.source_map.data['map_colors'] = self.source_map.data['fill_map'];
        self.source_map.data['map_values'] = self.source_map.data[attr_name];
        
        step = abs(edges[1] - edges[0])
        self.reset_sliders(step, edges)
        
    
    def add_slider_button_callback(self):
        if self.sliders_count == self.max_sliders_count:
            return 
        
        attr_name = self.select_attr.value
        hist, edges = self.hists_dict[attr_name]
        
        slider = Slider(start=edges[0], end=edges[-1], value=edges[0], step=abs(edges[1] - edges[0]))
        slider.on_change('value', self.sliders_hist_callback)
        slider.on_change('value_throttled', self.sliders_map_callback)
        
        self.sliders.append(slider)
        self.sliders_column.children.append(slider)
        self.sliders_count+=1
       
    
    def set_sliders(self, step, edges, values):
        for i, sl in enumerate(self.sliders):
            sl.start = edges[0]
            sl.end = edges[-1]
            sl.step = step
            if len(values) > i:
                sl.value = values[i]
            else:
                sl.value = edges[0]            
        
    def confirm_button_callback(self):
        attr_name = self.select_attr.value
        
        save_name = attr_name + self.attr_suff.value
        
        self.df[save_name] = self.source_map.data['map_colors']
        
       
        if save_name not in self.attr_tab_dict.keys():
            l = self.selection_group.labels
            if l[0] == 'Selection group':
                l.clear()

            l.append(save_name)
            self.selection_group.labels = l

            self.confirmed = False
            self.selection_group.active = len(l) - 1
            self.confirmed = True
            self.tabs_list.append(save_name)
        
        self.attr_tab_dict[save_name] = {'split': self.sliders_split_bins,
                                 'hist': self.histo_bins_count,
                                 'attr': attr_name
                               }

        
    def change_selection_callback(self, attr, old, new):
        
        if not self.confirmed:
            return
        
        
        key = self.tabs_list[new]
        self.sliders_split_bins = self.attr_tab_dict[key]['split']
        self.histo_bins_count = self.attr_tab_dict[key]['hist']
        attr_name = self.attr_tab_dict[key]['attr']
        self.select_attr.value = attr_name
        
        keep_values = self.sliders_split_bins
        self.bins_input_callback(None, None, str(self.histo_bins_count))
        self.attr_suff.value = key[len(attr_name):]
        self.sliders_split_bins = keep_values
        
        self.bins_input.value =  str(self.histo_bins_count)
        
        hist, edges = self.hists_dict[attr_name]
        self.set_sliders(abs(edges[1] - edges[0]), edges, self.sliders_split_bins)
        self.sliders_map_callback(None, None, None)
        
    def create_control_panel(self, mode, callback):
        self.select_attr_cmap = Select(title="Cmap:", value=self.attr_colors[0], options=self.attr_colors)
        self.select_attr_cmap.on_change('value', self.select_attr_cmap_callback)
        
        self.select_attr = Select(title="Attribute:", value=self.attributes[0], options=self.attributes)
        self.select_attr.on_change('value', self.select_attr_callback)
        
        
        self.bins_input = TextInput(value=str(self.histo_bins_count), title="Bins count:")
        self.bins_input.on_change('value', self.bins_input_callback)
        
        
        self.attr_suff = TextInput(value="_h", title="Attribute Suffix")
        
        
        self.create_slider_btn = Button(label="Add slider", button_type="success")
        self.create_slider_btn.on_click(self.add_slider_button_callback)
        
        
        self.confirm_split = Button(label="Confirm", button_type="success")
        self.confirm_split.on_click(self.confirm_button_callback)

        
        
        self.selection_group = RadioButtonGroup(labels=['Selection group'], active=0)
        self.selection_group.on_change('active', self.change_selection_callback)
        
            
        self.sliders = []
        start_hist, start_edges = self.hists_dict[self.attributes[0]]
        for i in range(self.sliders_count):
            slider = Slider(start=start_edges[0], end=start_edges[-1],
                        value=start_edges[0], step=abs(start_edges[1] - start_edges[0]))
            slider.on_change('value', self.sliders_hist_callback)
            slider.on_change('value_throttled', self.sliders_map_callback)
            self.sliders.append(slider)
            
        self.sliders_column = column(*self.sliders)

        elements = []

        if mode == 'server':
            elements.append(self.__create_save_panel())


        elements.extend([self.select_attr_cmap, self.select_attr,
                      self.bins_input, self.attr_suff, self.sliders_column, self.create_slider_btn, self.confirm_split])

        if mode == 'embed' and callback is not None:
            next_step_btn = Button(label="Next step", button_type="success")
            next_step_btn.on_click(partial(callback, "hist"))
            elements.append(next_step_btn)
        
        return column(*elements)





def hist_cluster_selector(df, cols, map_cmaps=None, max_sliders_count=7, 
                          hist_cmap=None, hist_size=(400, 350), map_size=(350, 350), 
                          mode='notebook', save_name='result_df.csv', callback=None):

    """
    Creates interactive tool for manual clusterization by lby histograms with saving into the source dataframe
   
    Parameters
    ----------
    Input:
        df - pandas.DataFrame, Table of data with attributes
        
        cols - list of strings; Names of columns (attributes) in "df" to be plotted
    
        map_cmaps - dict {cmap_name: cmap} or None; Dictionary with colormaps to be used for visualizing maps of attributes

        max_sliders_count - int; Max number of selections on histogram, e.g 7 is set by default
            
        hist_cmap - bokeh.models.palettes; Colormap for histogram
            
        hist_size - tuple of two ints; Means size of histogram in the figure, e.g. (500, 500) is set by default.
            
        map_size - tuple of two ints; Means size of each map in the figure, e.g. (500, 500) is set by default.

        mode - 'notebook', 'server', 'embed'; Use mode 'notebook' in jupyter notebook like environment.
                                             Use mode 'server' to launch bokeh app (bokeh serve --show script.py)
                                             Mode 'embed' is only for ClusterPipeline.

        save_name - path to save source dataframe, e.g. 'result_df.csv'is set by default

    Output:
        in mode 'notebook': fig: bokeh.models.layout. This is plotted via bokeh.plotting.show
        in mode 'server': None
        in mode 'embed': returns _HistClusterSelector() object. Use _HistClusterSelector().layout to 
        get bokeh layout and embed tool to another bokeh layout														
    """
    
    if map_cmaps is None:
        map_cmaps = {'Turbo': Turbo256, 'Viridis': Viridis256, 'Magma': Magma256, 
                     'Plasma': Plasma256, 'Cividis': Cividis256, 'Inferno': Inferno256, 'Greys': Greys256}
    if hist_cmap is None:
        hist_cmap = Turbo256

    if mode == 'notebook':
        output_notebook()
        def launch(doc):        
            cs = _HistClusterSelector(doc, df, cols, map_cmaps, max_sliders_count, 
                                     hist_cmap, hist_size, map_size, mode, callback, save_name)
        return launch
    elif mode == 'server':
        output_file('out.html')
        doc = curdoc()
        cs = _HistClusterSelector(doc, df, cols, map_cmaps, max_sliders_count, 
                                     hist_cmap, hist_size, map_size, mode, callback, save_name)

    elif mode == 'embed':
        cs = _HistClusterSelector(None, df, cols, map_cmaps, max_sliders_count, 
                                     hist_cmap, hist_size, map_size, mode, callback, save_name)
        return cs
    else:
        print('Unknown launch mode. Available modes are notebook, server, embed')

    return None


class _ClusterPipeline:
    def __init__(self, doc, df, map_cols, init_hist_selector=None, init_cluster_selector=None,
        hist_kwargs=None, cluster_kwargs=None, save_name='result_df.csv', img_path='out_pics'):

        if doc is None:
            self.doc = curdoc()

        self.df = df
        self.map_cols = map_cols
        self.init_cluster_selector = init_cluster_selector
        self.cluster_kwargs = cluster_kwargs
        self.save_name = save_name
        self.img_path = img_path


        self.__save_maps()

        if init_hist_selector is None:
            hs = hist_cluster_selector(self.df, list(self.df.columns))
        else:
            hs = hist_cluster_selector(**init_hist_selector(self.df, hist_kwargs), mode='embed', callback=self.__main_loop)

        self.layout = column(spacing=100)
        self.layout.children.append(hs.layout)



        save_btn = Button(label="Save result dataframe", button_type="success")
        save_btn.on_click(self.__save_dataframe)

        download_btn = Button(label="Download result dataframe", button_type="success")
        download_btn.js_on_click(CustomJS(args=dict(source=ColumnDataSource(self.df)),
                                    code=open(join(dirname(__file__), 'download_df.js')).read()))

        self.save_panel = row(save_btn, download_btn)

        self.doc.add_root(column(self.save_panel, self.layout, spacing=30))


    def __save_dataframe(self):
        self.df.to_csv(join(dirname(__file__), self.save_name))

    def __save_maps(self):

        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        layout = plot_maps(self.df, kdims=['X', 'Y'], vdims=self.map_cols,
        ncols=3, y_sampl=1, raster_height=65, raster_width=180)

        hv.renderer('matplotlib').save(layout, self.img_path + '/maps', fmt='png')


    def __main_loop(self, stage):
        if stage == 'hist': 
            if len(self.layout.children) != 1:
                self.layout.children.pop()
                return
            
            if self.init_cluster_selector is None:
                cs = cluster_selector(self.df, list(self.df.columns), k_map=['X', 'Y'], k_emb=['F1', 'F2'])
            else:
                cs = cluster_selector(**self.init_cluster_selector(self.df, self.cluster_kwargs), mode='embed', callback=self.__main_loop)

            self.layout.children.append(cs.layout)

def create_cluster_pipeline(df, map_cols, init_hist_selector=None, mode='server',
    init_cluster_selector=None, hist_kwargs=None, cluster_kwargs=None, save_name='result_df.csv', img_path='out_pics'):

    """
    Creates interactive tool for manual clusterization by lby histograms with saving into the source dataframe
   
    Parameters
    ----------
    Input:
        df - pandas.DataFrame, Table of data with attributes

        init_hist_selector - function with signature func(df, hist_kwargs) --> dict; Returns dict with args to 'hist_cluster_selector' tool

        init_cluster_selector - function with signature func(df, cluster_kwargs) --> dict; Returns dict with args to 'cluster_selector' tool
                                'df' is the result df from  'hist_cluster_selector' tool

        hist_kwargs - dict; Arguments to  'init_hist_selector'

        cluster_kwargs - dict; Arguments to  'init_cluster_selector'

        save_name - path to save source dataframe, e.g. 'result_df.csv'is set by default

    Output:
            None
    """

    if mode == 'notebook':
        def launch(doc):        
            _ClusterPipeline(doc, df, map_cols, init_hist_selector, init_cluster_selector, hist_kwargs, cluster_kwargs, save_name, img_path)
        return launch

    _ClusterPipeline(None, df, map_cols, init_hist_selector, init_cluster_selector, hist_kwargs, cluster_kwargs, save_name, img_path)