<template>
  <div id="app">
    <h1 class="header">Advanced Vietnamese Text Summarizer</h1>
    <p class="author"><center> Quang Hoang - Binh Dao - Ha Pham </center></p>
    <div id="nav">
      <router-link to="/">Extraction</router-link> |
      <router-link to="/abstractive">Abstraction</router-link>
    </div>
    <router-view/>
  </div>
</template>

<style>

#app .header {
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-weight: 1200;
  font-size:1cm;
  letter-spacing: +0.37px;
  color: rgb(255, 255, 255);
  background-image: linear-gradient(
    -225deg,
    #a445b2 0%,
    #d41872 52%,
    #ff0066 100%
  );
}

#app .author {
  text-align: center;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-weight: 100;
  font-size:10;
  letter-spacing: +0.37px;
  color: rgb(80, 0, 0)
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {
  padding: 0px;
}

#nav a {
  font-weight: bold;
  color: #110344;
}

#nav a.router-link-exact-active {
  color: #f09433;
  padding: 0cm;
}

</style>
