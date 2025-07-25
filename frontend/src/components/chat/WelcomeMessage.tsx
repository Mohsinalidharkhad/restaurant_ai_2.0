'use client'

import React from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Sparkles, Menu, Users, Clock } from 'lucide-react'

export default function WelcomeMessage() {
  return (
    <div className="px-6 py-4">
      <Card className="restaurant-gradient text-white">
        <CardContent className="p-6 text-center">
          <div className="flex items-center justify-center mb-4">
            <Sparkles className="w-8 h-8 mr-2" />
            <h2 className="text-2xl font-bold">Welcome to Silk Route Eatery!</h2>
            <Sparkles className="w-8 h-8 ml-2" />
          </div>
          
          <p className="text-lg mb-6">
            I'm your personal dining assistant, ready to help you explore our delicious menu.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
              <Menu className="w-6 h-6 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">Browse Freely</h3>
              <p className="text-sm">Ask about dishes, ingredients, and dietary options</p>
            </div>
            
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
              <Users className="w-6 h-6 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">No Registration Required</h3>
              <p className="text-sm">Explore our menu without any signup</p>
            </div>
            
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
              <Clock className="w-6 h-6 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">Quick Responses</h3>
              <p className="text-sm">Get instant recommendations and information</p>
            </div>
          </div>
          
          <p className="text-sm opacity-90">
            <strong>Ready to order or make a reservation?</strong> I'll help you register when you're ready!
          </p>
        </CardContent>
      </Card>
    </div>
  )
} 