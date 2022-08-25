/**
 * @fileoverview Type definitions for react-plotly.js relayout event handling.
 */

// Disables lint error since plotly dependency ends in '.js'.
// tslint:disable-next-line:ban-malformed-import-paths
import * as plotly from 'plotly.js';
import * as react from 'react';


/**
 * Plotly.js figure Object type interface.
 * Adapted to our use case and to react-plotly.js
 * Inferred from the original @types/plotly.js.
 */
export interface Figure {
  data: plotly.Data[];
  layout: Partial<plotly.Layout>;
  frames: plotly.Frame[]|null;
}


/**
 * react-plotly.js type interface for Plot parameters.
 * Adapted to our use case and to react-plotly.js
 * Inferred from the original @types/plotly.js.
 */
export interface PlotParams {
  data: plotly.Data[];
  layout: Partial<plotly.Layout>;
  frames?: plotly.Frame[];
  config?: Partial<plotly.Config>;

  /**
   * Callback executed after plot is initialized.
   * @param figure Object with three keys corresponding to input props: data,
   *     layout and frames.
   * @param graphDiv Reference to the DOM node into which the figure was
   *     rendered.
   */
  onInitialized?:
      ((figure: Readonly<Figure>, graphDiv: Readonly<HTMLElement>) => void);

  /**
   * Callback executed when when a plot is updated due to new data or layout, or
   * when user interacts with a plot.
   * @param figure Object with three keys corresponding to input props: data,
   *     layout and frames.
   * @param graphDiv Reference to the DOM node into which the figure was
   *     rendered.
   */
  onUpdate?:
      ((figure: Readonly<Figure>, graphDiv: Readonly<HTMLElement>) => void);

  /**
   * used to style the <div> into which the plot is rendered
   */
  style?: react.CSSProperties;

  /**
   * When true, adds a call to Plotly.Plot.resize() as a window.resize event
   * handler
   */
  useResizeHandler?: boolean;

  /**
   * Plotly event type definition
   */
  onRelayout?: ((event: Readonly<plotly.PlotRelayoutEvent>) => void);
}

/**
 * react-plotly.js type interface for Plot object.
 */
export class Plot extends react.PureComponent<PlotParams> {}