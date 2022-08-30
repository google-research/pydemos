/**
 * @fileoverview Imports Svelte Component and creates connection
 * with Streamlit.
 */

import ImageGallery from './ImageCarousel.svelte';
import {WithStreamlitConnection} from './streamlit';


// "WithStreamlitConnection" is a wrapper component. It bootstraps the
// connection between the component and the Streamlit app, and handles
// passing arguments from Python -> Component.
// tslint:disable-next-line:no-unused-variable
const imageGallery = new WithStreamlitConnection({
  target: document.body,
  props: {
    component: ImageGallery,
    spreadArgs: true,
  },
});

// Disables lint error since default export is necessary for Streamlit.
// tslint:disable-next-line:no-default-export
export default ImageGallery;
