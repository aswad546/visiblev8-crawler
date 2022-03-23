import {createRouter, createWebHistory} from 'vue-router'

import Index from '@/components/Index.vue'
import Result from '@/components/Result.vue'
import About from '@/components/About.vue'

// add router

const routes = [
    {path: '/', name: 'Index', component: Index},
    {path: '/about', name: 'About', component: About},
    {path: '/result', name: 'Result', component: Result},


]


const router = createRouter({
    history: createWebHistory(),
    routes
  })

export default router