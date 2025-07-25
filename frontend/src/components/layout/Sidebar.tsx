'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { UserInfo } from '@/lib/types'
import { User, Phone, CheckCircle, Info } from 'lucide-react'

interface SidebarProps {
  userInfo: UserInfo
}

export default function Sidebar({ userInfo }: SidebarProps) {
  const getStatusInfo = () => {
    if (userInfo.isRegistered) {
      return {
        text: "Registered",
        icon: <CheckCircle className="w-4 h-4" />,
        className: "status-indicator status-registered",
        description: "Full access to all features"
      }
    } else if (userInfo.hasPhone) {
      return {
        text: "Phone Verified",
        icon: <Phone className="w-4 h-4" />,
        className: "status-indicator status-phone-verified",
        description: "Ready for reservations"
      }
    } else {
      return {
        text: "Browse Mode",
        icon: <Info className="w-4 h-4" />,
        className: "status-indicator status-not-registered",
        description: "Menu browsing available"
      }
    }
  }

  const statusInfo = getStatusInfo()

  return (
    <div className="w-80 h-screen bg-gray-50 border-r border-gray-200 p-4 flex flex-col overflow-hidden">
      <Card className="mb-6 flex-shrink-0">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <User className="w-5 h-5" />
            User Information
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Status */}
          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 block">Status</label>
            <div className={statusInfo.className}>
              {statusInfo.icon}
              <span className="ml-1">{statusInfo.text}</span>
            </div>
            <p className="text-xs text-gray-500 mt-1">{statusInfo.description}</p>
          </div>

          {/* User Details */}
          {(userInfo.isRegistered || userInfo.hasPhone) && (
            <div className="space-y-3">
              {userInfo.name && (
                <div>
                  <label className="text-sm font-medium text-gray-700">Name</label>
                  <p className="text-sm text-gray-900 bg-white px-3 py-2 rounded border">
                    {userInfo.name}
                  </p>
                </div>
              )}
              
              {userInfo.phone && (
                <div>
                  <label className="text-sm font-medium text-gray-700">Mobile Number</label>
                  <p className="text-sm text-gray-900 bg-white px-3 py-2 rounded border">
                    {userInfo.phone}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Information Panel */}
          <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
            <h4 className="text-sm font-medium text-blue-900 mb-2">What you can do:</h4>
            <ul className="text-xs text-blue-800 space-y-1">
              <li>• Browse menu and get recommendations</li>
              <li>• Ask about ingredients and dietary options</li>
              {userInfo.hasPhone && (
                <li>• Make and manage reservations</li>
              )}
              {userInfo.isRegistered && (
                <>
                  <li>• Place orders</li>
                  <li>• View billing information</li>
                </>
              )}
            </ul>
          </div>

          {!userInfo.isRegistered && (
            <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
              <h4 className="text-sm font-medium text-yellow-900 mb-1">Need to register?</h4>
              <p className="text-xs text-yellow-800">
                Registration is only needed when you're ready to place an order or make a reservation.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* App Info - stays at bottom */}
      <div className="mt-auto flex-shrink-0 text-center text-xs text-gray-500">
        <p>Restaurant Assistant v1.0</p>
      </div>
    </div>
  )
} 