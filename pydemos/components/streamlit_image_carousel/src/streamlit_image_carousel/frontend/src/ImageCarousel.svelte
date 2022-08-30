<script lang="ts">
  import { Streamlit, setStreamlitLifecycle } from "./streamlit";

  setStreamlitLifecycle();

  // Access arguments sent from Streamlit
  export let image_list: string[];
  export let scroller_height: number;

  let selectedImageUrl: string;

  function onClickEventHandler(event: any) {
    selectedImageUrl = event.srcElement.currentSrc;
    Streamlit.setComponentValue(selectedImageUrl);
  }
</script>

<div class="scroller">
  {#each image_list as imageUrl}
    <img
      src={imageUrl}
      id={imageUrl}
      alt="Image URL unavailable"
      style="height: {scroller_height}px;"
      on:click={onClickEventHandler}
    />
  {/each}
</div>

<style>
  .scroller {
    min-height: 100px;
    overflow-x: scroll;
    overflow-y: hidden;
    white-space: nowrap;
  }

  img {
    display: inline-block;
    padding: 1%;
    border-radius: 20px;
    opacity: 0.9;
    transition: all 0.2s;
  }

  img:hover {
    opacity: 1;
    transform: scale(1.05);
    cursor: pointer;
  }

  .scroller::-webkit-scrollbar-track {
    -webkit-box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
    box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    background-color: #f5f5f5;
  }

  .scroller::-webkit-scrollbar {
    width: 7px;
    background-color: #f5f5f5;
  }

  .scroller::-webkit-scrollbar-thumb {
    border-radius: 10px;
    -webkit-box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
    box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
    background-color: #555;
  }
</style>
