<template>
  <div id="app">
    <h1>Text Summarizer</h1>
    <textarea v-model="inputText" placeholder="请在此输入需要摘要的文本..." rows="10" cols="50"></textarea><br>
    <button @click="submitText">生成式摘要</button>
    <button @click="submitTextAlt">抽取式摘要</button> <!-- 新按钮 -->

    <div class="summary-container" v-if="summary">
      <h2>Summary</h2>
      <div>{{ summary }}</div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'App',
  data() {
    return {
      inputText: '',
      summary: ''
    }
  },
  methods: {
    submitText() {
      axios.post('http://localhost:5000/summarize', { text: this.inputText })
        .then(response => {
          this.summary = response.data.summary;
        })
        .catch(error => {
          console.error(error);
        });
    },
    submitTextAlt() { // 新方法
      axios.post('http://localhost:5000/summarize_alt', { text: this.inputText })
        .then(response => {
          this.summary = response.data.summary;
        })
        .catch(error => {
          console.error(error);
        });
    }
  }
}
</script>

<style>
body {
  font-family: 'Arial', sans-serif;
  background-color: #f0f3f5; /* 更柔和的背景色 */
  margin: 0;
  padding: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

#app {
  width: 80%;
  max-width: 600px;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); /* 更柔和的阴影 */
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

textarea {
  width: 100%;
  padding: 10px;
  margin-bottom: 20px; /* 增加间距 */
  border-radius: 4px;
  border: 1px solid #b0bec5; /* 更明显的边框颜色 */
  box-sizing: border-box;
  font-family: 'Arial', sans-serif;
}

button {
  background-color: #42a5f5; /* 更明亮的蓝色 */
  color: white;
  padding: 10px 15px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  margin: 5px;
  transition: background-color 0.3s; /* 平滑的背景色过渡 */
}

button:hover {
  background-color: #1e88e5; /* 悬停时更深的蓝色 */
}

button:nth-of-type(1) {
  background-color: #ffca28; /* 第一个按钮的颜色 */
}

button:nth-of-type(1):hover {
  background-color: #ffb300; /* 第一个按钮悬停的颜色 */
}

h1 {
  color: #424242; /* 更深的标题颜色 */
}

h2 {
  color: #424242;
  margin-bottom: 10px; /* 增加标题下的间距 */
}

.summary-container {
  margin-top: 20px;
  padding: 15px; /* 更多的内部空间 */
  background-color: #e3f2fd; /* 更亮的背景色 */
  border-radius: 4px;
  border: 1px solid #90caf9; /* 更明亮的边框颜色 */
  width: 100%; /* 确保摘要容器宽度与 textarea 相同 */
  box-sizing: border-box;
}

@media (max-width: 600px) {
  #app {
    width: 95%;
    margin: 10px;
  }

  textarea, button {
    font-size: 14px;
  }
}
</style>

