import { useNavigate } from 'react-router-dom';
import { Button } from "@/components/ui/button"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"

export default function Home() {
    const navigate = useNavigate();

  const handleClick = () => {
    navigate('/enrollment'); // Navigate to the /dashboard route
  };
  return (
       <div className="min-h-screen bg-white dark:bg-black">
      {/* Header */}
      <header className="border-b bg-white dark:bg-black">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-black dark:text-white">
                VirtualSecure
              </h1>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          {/* Hero Section */}
          <div className="mb-16">
            <div className="mx-auto mb-6 p-4 bg-gray-100 dark:bg-gray-800 rounded-full w-fit">
              <div className="h-12 w-12 bg-black dark:bg-white rounded"></div>
            </div>
            <h1 className="text-4xl font-bold text-black dark:text-white mb-4">
              VirtualSecure
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
              Advanced virtual device app with face recognition, privacy
              protection, and streaming safety features.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  size="lg"
                  className="w-full sm:w-auto bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200" onClick={handleClick}
                >
                  Start Enrollment
                </Button>
              <Button
                variant="outline"
                size="lg"
                disabled
                className="w-full sm:w-auto border-gray-300 text-gray-400 dark:border-gray-600 dark:text-gray-500"
              >
                Go Live (Coming Soon)
              </Button>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            <Card className="shadow-lg hover:shadow-xl transition-shadow border border-gray-200 dark:border-gray-700">
              <CardHeader>
                <div className="mx-auto mb-3 p-3 bg-gray-100 dark:bg-gray-800 rounded-full w-fit">
                  <div className="h-6 w-6 bg-black dark:bg-white rounded"></div>
                </div>
                <CardTitle className="text-black dark:text-white">
                  Whitelist Enrollment
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Secure face capture and enrollment system for trusted
                  identification
                </CardDescription>
              </CardHeader>
              <CardContent>
                  <Button
                    variant="outline"
                    className="w-full border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                  >
                    Get Started
                  </Button>
              </CardContent>
            </Card>

            <Card className="shadow-lg hover:shadow-xl transition-shadow border border-gray-200 dark:border-gray-700">
              <CardHeader>
                <div className="mx-auto mb-3 p-3 bg-gray-100 dark:bg-gray-800 rounded-full w-fit">
                  <div className="h-6 w-6 bg-black dark:bg-white rounded"></div>
                </div>
                <CardTitle className="text-black dark:text-white">
                  Live Streaming
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Stream with real-time face blurring and privacy protection
                  features
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  variant="outline"
                  className="w-full border-gray-300 text-gray-400 dark:border-gray-600 dark:text-gray-500"
                  disabled
                >
                  Coming Soon
                </Button>
              </CardContent>
            </Card>

            <Card className="shadow-lg hover:shadow-xl transition-shadow border border-gray-200 dark:border-gray-700">
              <CardHeader>
                <div className="mx-auto mb-3 p-3 bg-gray-100 dark:bg-gray-800 rounded-full w-fit">
                  <div className="h-6 w-6 bg-black dark:bg-white rounded"></div>
                </div>
                <CardTitle className="text-black dark:text-white">
                  Safety Analytics
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Post-stream safety scores and comprehensive privacy reports
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  variant="outline"
                  className="w-full border-gray-300 text-gray-400 dark:border-gray-600 dark:text-gray-500"
                  disabled
                >
                  Coming Soon
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* How it Works */}
          <div className="text-left">
            <h2 className="text-2xl font-bold text-center mb-8 text-black dark:text-white">
              How it Works
            </h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="w-8 h-8 bg-black dark:bg-white text-white dark:text-black rounded-full flex items-center justify-center text-sm font-bold mx-auto mb-3">
                  1
                </div>
                <h3 className="font-semibold mb-2 text-black dark:text-white">
                  Enroll Your Face
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Capture your face securely to join the whitelist for clear
                  visibility during streams.
                </p>
              </div>
              <div className="text-center">
                <div className="w-8 h-8 bg-black dark:bg-white text-white dark:text-black rounded-full flex items-center justify-center text-sm font-bold mx-auto mb-3">
                  2
                </div>
                <h3 className="font-semibold mb-2 text-black dark:text-white">
                  Stream Safely
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Non-whitelisted faces are automatically blurred to protect
                  privacy and maintain safety.
                </p>
              </div>
              <div className="text-center">
                <div className="w-8 h-8 bg-black dark:bg-white text-white dark:text-black rounded-full flex items-center justify-center text-sm font-bold mx-auto mb-3">
                  3
                </div>
                <h3 className="font-semibold mb-2 text-black dark:text-white">
                  Monitor Safety
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Get detailed safety scores and analytics after each streaming
                  session.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t bg-white dark:bg-black mt-16">
        <div className="container mx-auto px-4 py-8 text-center text-sm text-gray-600 dark:text-gray-400">
          <p>© 2025 VirtualSecure. Built for TikTok TechJam 2025.</p>
        </div>
      </footer>
    </div>
  )
}

