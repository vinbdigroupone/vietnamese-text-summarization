<template>
  <!-- eslint-disable max-len -->
  <div>
    <link
      rel="stylesheet"
      href="https://use.fontawesome.com/releases/v5.8.2/css/all.css"
      integrity="sha384-oS3vJWv+0UjzBfQzYUhtDYW+Pj2yciDJxpsK1OYPAYjqT085Qq/1cq5FLXAZQ7Ay"
      crossorigin="anonymous"
    >

    <center style="margin-top: 20px">
        <toggle-button
          class="mb-3"
          v-model="from_raw_text"
          :value="false"
          color =#f09433
          :sync="true"
          :width="60"
          :labels="{checked:'TEXT', unchecked: 'URL'}"
        />
    </center>

    <div>
      <div class="row" style="height: 100%;display: flex;justify-content: space-around; margin-top: 20px; margin-bottom: 20px">
        <div class='col' @submit="onSubmit">
          <textarea class="inputbox" v-model="text" placeholder='Enter text or URL'/>
        </div>
        <div class='col'>
          <textarea class="inputbox" v-model="answer" placeholder='Output' readonly/>
        </div>
      </div>
      <button class="gradient-fill background hover" @click="onSubmit">
          <!-- eslint-disable-next-line vue/max-attributes-per-line -->
          <b-spinner v-if="status == 'loading'" small :variant="'white'" label="Spinner"></b-spinner>
          <i class="fa fa-search"></i>
        </button>
    </div>

  </div>
</template>

<script>
import axios from 'axios';
import { ToggleButton } from 'vue-js-toggle-button';

export default {
  name: 'Extractive',
  components: {
    ToggleButton,
  },
  props: {
    api_endpoint_raw: {
      type: String,
      default: 'http://127.0.0.1:5000/raw',
    },
    api_endpoint_url: {
      type: String,
      default: 'http://127.0.0.1:5000/url',
    },
  },
  data() {
    return {
      text: '',
      from_raw_text: false,
      status: 'started',
      answer: '',
    };
  },
  methods: {
    onSubmit(evt) {
      evt.preventDefault();
      this.status = 'loading';
      let apiEndpoint = this.api_endpoint_raw;
      if (!this.from_raw_text) {
        apiEndpoint = this.api_endpoint_url;
      }
      axios
        .get(apiEndpoint, { params: { text: this.text } })
        .then((response) => {
          this.answer = response.data.answer;
          this.status = 'done';
        })
        .catch((error) => {
          console.error(error);
        });
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.textsummui {
  text-align: left;
  font-family: Inter, Inter UI, Inter-UI, SF Pro Display, SF UI Text,
    Helvetica Neue, Helvetica, Arial, sans-serif;
  font-weight: 400;
  letter-spacing: +0.37px;
  color: rgb(175, 175, 175);
}

.form-control:focus {
  border-color: #ae41a7 !important;
  box-shadow: 0 0 5px #ae41a7 !important;
}

.toggle {
  padding: 10px;
  color: linear-gradient(45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #bc1888 100%
  );
}

.inputbox {
  width: 500px;
  height: 500px;
  text-align: center;
}

.fa-search::before {
    content: "Summarize";
}

.instagram{
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-weight: 800;
  letter-spacing: +0.37px;
  color: rgb(255, 255, 255);
  background: #f09433;
  background: -moz-linear-gradient(
    45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #bc1888 100%
  );
  background: -webkit-linear-gradient(
    45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #806878 100%
  );
  background: linear-gradient(45deg,
    #f09433 0%,
    #e6683c 25%,
    #dc2743 50%,
    #cc2366 75%,
    #bc1888 100%
  );
  filter: progid:DXImageTransform.Microsoft.gradient( startColorstr=#f09433,
    endColorstr=#bc1888,
    GradientType=1
  );
}

.header {
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-weight: 800;
  letter-spacing: +0.37px;
  color: rgb(255, 255, 255);
  background-image: linear-gradient(
    -225deg,
    #a445b2 0%,
    #d41872 52%,
    #ff0066 100%
  );
}

.gradient-fill {
  background-image: linear-gradient(
    -225deg,
    #a445b2 0%,
    #d41872 52%,
    #ff0066 100%
  );
}

.gradient-fill.background {
  background-size: 250% auto;
  border: medium none currentcolor;
  border-image: none 100% 1 0 stretch;
  transition-delay: 0s, 0s, 0s, 0s, 0s, 0s;
  transition-duration: 0.5s, 0.2s, 0.2s, 0.2s, 0.2s, 0.2s;
  transition-property: background-position, transform, box-shadow, border,
    transform, box-shadow;
  transition-timing-function: ease-out, ease, ease, ease, ease, ease;
  color: white;
  font-weight: bold;
  border-radius: 5px;
  margin-top: 10px;
}

span.gradient-fill {
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 20px;
  font-weight: 700;
  line-height: 2.5;
}
</style>
