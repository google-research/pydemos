/**
 * Streamlit Component to handle relayout events in Plotly.
 *
 * The component declares a connection with Streamlit.
 * On receiving a relayout event from Plotly, returns a JSON that, parsed,
 * contains a list of objects with the coordinates of each shapes drawn
 * on the plot.
 *
 * Format of object:
 *      {
 *          x0: number (value of smallest x coord),
 *          x1: number (value of largest x coord),
 *          y0: number (value of smallest y coord),
 *          y1: number (value of largest y coord)
 *      }
 */

import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from 'streamlit-component-lib';
import React, {ReactNode} from 'react';
import * as reactPlotly from './react_plotly_types';
// Disables lint error since plotly and react-plotly dependencies end in '.js'.
// tslint:disable:ban-malformed-import-paths
import Plot from 'react-plotly.js';
import {Shape} from 'plotly.js';
// tslint:enable:ban-malformed-import-paths

interface State {
  data: reactPlotly.Figure['data'];
  layout: reactPlotly.Figure['layout'];
  config: reactPlotly.PlotParams['config'];
  frames: reactPlotly.Figure['frames'];
}

class PlotlyRelayoutEventHandler extends StreamlitComponentBase<State> {
  state: State = {data: [], layout: {}, config: {}, frames: []};

  render = (): ReactNode => {
    // Pull Plotly object from args and parse
    const plotObj = JSON.parse(this.props.args['plot_obj']);

    const overrideHeight = this.props.args['override_height'];
    const overrideWidth = this.props.args['override_width'];

    // Important to set for Streamlit's own component rendering
    Streamlit.setFrameHeight(overrideHeight);

    return (
      <Plot
        // Plot props.
        data={this.state.data}
        layout={this.selectOnlyLastShape(this.state.layout)}
        config={this.state.config}
        frames={this.state.frames}
        // Set plot state on initialization and keep it updated on changes.
        onInitialized={() => this.setState(plotObj)}
        onUpdate={figure => this.setState(figure)}
        // Relayout event handler.
        onRelayout={(data: Partial<reactPlotly.PlotParams['layout']>) => {
          this.relayoutEventHandler(data);
        }}
        // Styling.
        style={{height: overrideHeight, width: overrideWidth}}
        useResizeHandler={true}
        className="stPlot"
      />
    );
  };

  /** Relayout event handler for plot. */
  private relayoutEventHandler(data: Partial<reactPlotly.PlotParams['layout']>) {
    // Build array of boxes to return
    const eventData: object[] = [];

    // If there is return data, iterate through it and add each shape's
    // coordinates to the eventData array that we will return at the end.
    if (data.shapes){
        Object.values(data.shapes).forEach((shape: Partial<Shape>) => {
          eventData.push({
            x0: shape.x0,
            x1: shape.x1,
            y0: shape.y0,
            y1: shape.y1,
          });
        });
    }

    // Return array as JSON to Streamlit
    Streamlit.setComponentValue(JSON.stringify(eventData));
    this.render();
  }

  private selectOnlyLastShape(layout: reactPlotly.PlotParams['layout']) {
    if (layout.shapes) {
      const layoutShapesLength = Object.keys(layout.shapes).length;
      // Keep only last drawn shape.
      layout.shapes = layout.shapes[layoutShapesLength - 1] as Array<
        Partial<Shape>
      >;
    }

    return layout;
  }
}

// Disables lint error since default export is necessary for Streamlit.
// tslint:disable-next-line:no-default-export
export default withStreamlitConnection(PlotlyRelayoutEventHandler);
