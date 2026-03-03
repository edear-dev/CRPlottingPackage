import base64
import matplotlib.pyplot as plt
import numpy as np

from .CRAB_data import CRdata

class CRplotter:
    def __init__(self, CRdatasets:list[CRdata], dtype:str, labels:list[str]=None, rows=1, columns=1, sharex:bool=True):
        if not isinstance(CRdatasets, list):
            raise ValueError('Please input the data as a list dude!!!')
        self.CRsets = CRdatasets
        datasets = []
        units = set()
        flux_units = set()
        for dset in self.CRsets:
            datasets.append(dset.data)
            units.add(dset.unit)
            flux_units.add(dset.flux_unit)
        if len(units) == 1:
            self.unit = units.pop()
        else:
            raise ValueError('All units should match')
        if len(flux_units) == 1:
            self.flux_unit = flux_units.pop()
        else:
            raise ValueError('All flux units should match')
        self.datasets = datasets 
        self.dtype = dtype
        self.labels = labels or [None] * len(datasets)
        self.rows = rows
        self.columns = columns
        self.fig, self.axes = plt.subplots(self.rows, self.columns, sharex=sharex)

    @classmethod
    def default_plot(cls, CRdata:list, dtype:str):
        labels = []
        for obj in CRdata:
            labels.append(obj.exp_name + ' ' + obj.element)
        plot = CRplotter(CRdata, dtype, labels=labels)
        plot.singlePlot()
        plot.setUp()
        plot.spruceitUp(title='Flux vs ' + dtype, xval=dtype, xunit=plot.unit, yunit=plot.flux_unit)

    def setUp(self, xlog:bool=True, ylog:bool=True, xbounds:tuple=None, ybounds:tuple=None):
        ax = self.axes
        ax.set(xscale = 'log' if xlog else 'linear',
               yscale = 'log' if ylog else 'linear',
               xlim = xbounds or None,
               ylim = ybounds or None)
        
    def singlePlot(self, which:list[int]=None, colors:list[str]=None):
        ax, labels = self.axes, self.labels
        datasets = self.datasets
        colors = colors or [None] * len(datasets)
        if which is None:
            which = list(range(len(datasets)))
        datasets = [datasets[i] for i in which]
        labels = [labels[i] for i in which]
        for dataset, label, color in zip(datasets, labels, colors):
            x, y, yerr = dataset[:, 0], dataset[:, 1], dataset[:,2]
            ax.errorbar(x, y, yerr, fmt='.', label=label, color=color)

    def plotFit(self, which:int=0, color:str=None):
        ax, label = self.axes, self.labels[which]
        dataset = self.datasets[which]
        ax.plot(dataset[:, 0], dataset[:, 1], label=label, color=color)

    def plotLine(self, h_or_v:str, val, color=None, fmt:str='dashed', label=None):
        # going to use hlines/vlines to keep room for future features
        ax = self.axes
        if h_or_v=='h':
            ax.hlines(val, *ax.get_xlim(), colors=color, linestyles=fmt, label=label)
        if h_or_v=='v':
            ax.vlines(val, *ax.get_ylim(), colors=color, linestyles=fmt, label=label) 
        

    def multPlot(self, data_all:list[str]=None, xlog=True, ylog=True, xbounds=None, ybounds=None, colors:list[str]=None):
        axes, labels = self.axes, self.labels
        colors = colors or [None] * len(self.datasets)
        data_list = self.datasets
        plot_all = False
        if data_all is not None:
            dsets_all = []
            labelsall = []
            colorsall = []
            for name in data_all:
                i = labels.index(name)
                dsets_all.append(self.datasets[i])
                data_list.pop(i)
                labelsall.append(labels[i])
                labels.pop(i)
                colorsall.append(colors[i])
                colors.pop(i)
            plot_all = True
        for ax, data, label, color in zip(axes, data_list, labels, colors):
            x = data[:,0]
            y = data[:,1]
            if plot_all:
                for allset, labelall, colorall in zip(dsets_all, labelsall, colorsall):
                    ax.errorbar(allset[:,0], allset[:,1], yerr=allset[:,2], fmt='.', label=labelall, color=colorall)
            ax.errorbar(x, y, yerr=data[:,2], fmt='.', label=label, color=color)
            ax.set(xscale = 'log' if xlog else 'linear',
               yscale = 'log' if ylog else 'linear',
               xlim = xbounds or None,
               ylim = ybounds or None)
            if any(labels):
                ax.legend()

    def spruceitUp(self, title, xval, xunit='G', ylabel=None, yunit='G', yarea_unit='m', power=0, right=False, save=False):
        ax = self.axes
        if any(self.labels):
            ax.legend()
        ax.set_title(title, fontweight='bold', fontsize=15)
        xlabel, ylabel = CRplotter.createLabels(xval, xunit, yunit, yarea_unit, power)
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        if right:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        if save:
            CRplotter.savePlot(title)

    def spruceitUpMult(self, title, xval, xunit='G', ylabel=None, yunit='G', yarea_unit='m', power=0, right=False, save=False):
        axes = self.axes
        self.fig.suptitle(title, fontweight='bold', fontsize=15)
        for ax in axes:
            if right:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
        xlabel, ylabel = CRplotter.createLabels(xval, xunit, yunit, yarea_unit, power)
        self.fig.supxlabel(xlabel, fontweight='bold', fontsize=12)
        self.fig.supylabel(ylabel, fontweight='bold', fontsize=12)
        self.fig.tight_layout()
        self.fig.set_size_inches(6,8)
        if save:
            CRplotter.savePlot(title)
    
    @staticmethod
    def createLabels(xval, xunit, yunit, yarea_unit, power):
        xlabel = xval + f' ({xunit})'

        if yunit.strip() == '[(s m^2 sr GeV)^-1]':
            yunit = 'GeV'
        # generate ylabel 
        base = rf"Φ ({yarea_unit}$^{{-2}}$s$^{{-1}}$sr$^{{-1}}${yunit}$^{{-1}}$)"
        if power == 0:
            ylabel = base
        else:
            ylabel = (
                rf"Φ {xval[0].upper()}$^{{{power}}}$"
                rf"({yarea_unit}$^{{-2}}$s$^{{-1}}$sr$^{{-1}}${yunit}"
                rf"$^{{{round(power-1,2)}}}$)"
                )
        return xlabel, ylabel
    
    @staticmethod
    def savePlot(title, artists:list=None, box:int=None):
        titlefinal = title.replace('/', '_')
        if box is None:
            plt.savefig(titlefinal + ' plot.png', transparent=True, bbox_inches='tight')
            #plt.savefig(titlefinal + ' plot.png', transparent=True, bbox_inches='tight', bbox_extra_artists=artists)
        else:
            plt.savefig(titlefinal + ' plot.png', transparent=True, bbox_inches=box, bbox_extra_artists=artists)

    def __str__(self):
        output = ''
        for label in self.labels:
            if label is None:
                return 'No labels provided'
            output = output + '\n-->' + label
        return 'Plotting' + output
    
    def __repr__(self):
        return f'CRplotter({self.CRsets},{self.labels},{self.rows},{self.columns})'
    

    ## going to have to adjust mulplot to new fig by fig setup