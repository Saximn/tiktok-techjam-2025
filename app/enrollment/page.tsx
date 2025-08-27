"use client";
import { useState } from "react";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CameraCapture } from "@/components/enrollment/CameraCapture";
import { FacePreview } from "@/components/enrollment/FacePreview";

export default function EnrollmentPage() {
  const [enrollmentStep, setEnrollmentStep] = useState<
    "intro" | "capture" | "form" | "preview" | "complete"
  >("intro");
  const [userName, setUserName] = useState("");
  const [capturedImage, setCapturedImage] = useState<string | null>(null);

  // Mock data for face preview
  const mockFaces = [
    {
      id: "1",
      name: "John Doe",
      image: "/placeholder-avatar.jpg",
      whitelisted: true,
    },
    {
      id: "2",
      name: "Unknown User",
      image: "/placeholder-avatar.jpg",
      whitelisted: false,
    },
  ];

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (userName.trim()) {
      setEnrollmentStep("preview");
    }
  };

  const handleComplete = () => {
    setEnrollmentStep("complete");
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
            <Badge
              variant="outline"
              className="text-sm border-black text-black dark:border-white dark:text-white"
            >
              Enrollment
            </Badge>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Progress Indicator */}
          <div className="mb-8">
            <div className="flex items-center justify-center space-x-4">
              {["intro", "capture", "form", "preview", "complete"].map(
                (step, index) => (
                  <div key={step} className="flex items-center">
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                    ${
                      enrollmentStep === step
                        ? "bg-black text-white dark:bg-white dark:text-black"
                        : [
                            "intro",
                            "capture",
                            "form",
                            "preview",
                            "complete",
                          ].indexOf(enrollmentStep) > index
                        ? "bg-gray-600 text-white dark:bg-gray-400 dark:text-black"
                        : "bg-gray-200 text-gray-500 dark:bg-gray-700 dark:text-gray-400"
                    }`}
                    >
                      {[
                        "intro",
                        "capture",
                        "form",
                        "preview",
                        "complete",
                      ].indexOf(enrollmentStep) > index
                        ? "✓"
                        : index + 1}
                    </div>
                    {index < 4 && (
                      <div
                        className={`w-12 h-0.5 mx-2 ${
                          [
                            "intro",
                            "capture",
                            "form",
                            "preview",
                            "complete",
                          ].indexOf(enrollmentStep) > index
                            ? "bg-gray-600 dark:bg-gray-400"
                            : "bg-gray-200 dark:bg-gray-700"
                        }`}
                      />
                    )}
                  </div>
                )
              )}
            </div>
            <div className="text-center mt-2">
              <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                {enrollmentStep === "intro"
                  ? "Welcome"
                  : enrollmentStep === "capture"
                  ? "Face Capture"
                  : enrollmentStep === "form"
                  ? "Personal Details"
                  : enrollmentStep === "preview"
                  ? "Review & Confirm"
                  : "Completed"}
              </p>
            </div>
          </div>

          {/* Introduction Step */}
          {enrollmentStep === "intro" && (
            <Card className="flex flex-col h-[600px] text-center shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-3xl mx-auto">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl text-black dark:text-white">
                  Welcome to VirtualSecure
                </CardTitle>
                <CardDescription className="text-lg max-w-2xl mx-auto text-gray-600 dark:text-gray-400">
                  Enroll in our whitelist system to enable face recognition and
                  privacy protection during your streams.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6 flex-1 flex flex-col justify-center">
                <div className="grid md:grid-cols-3 gap-4 text-left mx-auto">
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 ">
                    <h3 className="font-semibold mb-1 text-black dark:text-white">
                      Face Capture
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      We'll capture your face for secure identification
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                    <h3 className="font-semibold mb-1 text-black dark:text-white">
                      Privacy Protection
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Non-whitelisted faces will be automatically blurred
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                    <h3 className="font-semibold mb-1 text-black dark:text-white">
                      Trusted Access
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Only whitelisted individuals appear clearly
                    </p>
                  </div>
                </div>
              </CardContent>
              <div className="px-6 pb-6 mt-auto">
                <Button
                  onClick={() => setEnrollmentStep("capture")}
                  size="lg"
                  className="w-full mx-auto bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                >
                  Start Enrollment →
                </Button>
              </div>
            </Card>
          )}

          {/* Camera Capture Step */}
          {enrollmentStep === "capture" && (
            <Card className="flex flex-col h-[600px] text-center shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-3xl mx-auto">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl text-black dark:text-white">
                  Face Capture
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Position your face clearly in the camera and capture a
                  high-quality photo for enrollment.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <CameraCapture />
              </CardContent>
              <div className="flex justify-between px-6 pb-6 mt-auto">
                <Button
                  variant="outline"
                  onClick={() => setEnrollmentStep("intro")}
                  className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                >
                  Back
                </Button>
                <Button
                  onClick={() => setEnrollmentStep("form")}
                  disabled={!capturedImage}
                  className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                >
                  Continue →
                </Button>
              </div>
            </Card>
          )}

          {/* Form Step */}
          {enrollmentStep === "form" && (
            <Card className="shadow-xl border border-gray-200 dark:border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-black dark:text-white">
                  Personal Details
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Provide your information to complete the enrollment process.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleFormSubmit} className="space-y-6">
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2 text-black dark:text-white">
                        Full Name
                      </label>
                      <input
                        type="text"
                        value={userName}
                        onChange={(e) => setUserName(e.target.value)}
                        placeholder="Enter your full name"
                        className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-black focus:border-transparent bg-white dark:bg-black text-black dark:text-white"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2 text-black dark:text-white">
                        Email (Optional)
                      </label>
                      <input
                        type="email"
                        placeholder="your.email@example.com"
                        className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-black focus:border-transparent bg-white dark:bg-black text-black dark:text-white"
                      />
                    </div>
                    <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                      <div className="flex items-start gap-2">
                        <div className="h-5 w-5 bg-black dark:bg-white rounded mt-0.5"></div>
                        <div>
                          <h4 className="font-medium text-black dark:text-white">
                            Privacy Notice
                          </h4>
                          <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                            Your facial data is encrypted and stored securely.
                            It will only be used for identification during
                            streams.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-between">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setEnrollmentStep("capture")}
                      className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                    >
                      Back
                    </Button>
                    <Button
                      type="submit"
                      className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                    >
                      Review Enrollment →
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>
          )}

          {/* Preview Step */}
          {enrollmentStep === "preview" && (
            <Card className="shadow-xl border border-gray-200 dark:border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-black dark:text-white">
                  Review & Confirm
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Review your enrollment details and see how face recognition
                  will work.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-semibold mb-3 text-black dark:text-white">
                      Your Information
                    </h3>
                    <div className="space-y-2 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                      <p className="text-black dark:text-white">
                        <span className="font-medium">Name:</span> {userName}
                      </p>
                      <p className="text-black dark:text-white">
                        <span className="font-medium">Status:</span>{" "}
                        <Badge className="ml-2 bg-black text-white dark:bg-white dark:text-black">
                          Whitelisted
                        </Badge>
                      </p>
                      <p className="text-black dark:text-white">
                        <span className="font-medium">Enrollment Date:</span>{" "}
                        {new Date().toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-3 text-black dark:text-white">
                      Face Recognition Preview
                    </h3>
                    <FacePreview
                      faces={[
                        ...mockFaces,
                        {
                          id: "3",
                          name: userName,
                          image: capturedImage || "/placeholder-avatar.jpg",
                          whitelisted: true,
                        },
                      ]}
                    />
                  </div>
                </div>
                <div className="flex justify-between">
                  <Button
                    variant="outline"
                    onClick={() => setEnrollmentStep("form")}
                    className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                  >
                    Back
                  </Button>
                  <Button
                    onClick={handleComplete}
                    className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                  >
                    Complete Enrollment ✓
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Complete Step */}
          {enrollmentStep === "complete" && (
            <Card className="text-center shadow-xl border border-gray-200 dark:border-gray-700">
              <CardHeader className="pb-4">
                <div className="mx-auto mb-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-full w-fit">
                  <div className="h-8 w-8 bg-black dark:bg-white rounded-full flex items-center justify-center">
                    <span className="text-white dark:text-black text-lg">
                      ✓
                    </span>
                  </div>
                </div>
                <CardTitle className="text-2xl text-black dark:text-white">
                  Enrollment Complete!
                </CardTitle>
                <CardDescription className="text-lg max-w-2xl mx-auto text-gray-600 dark:text-gray-400">
                  You've been successfully added to the whitelist. You can now
                  start streaming with face recognition enabled.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                  <h3 className="font-semibold text-black dark:text-white mb-2">
                    What's Next?
                  </h3>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1 text-left max-w-md mx-auto">
                    <li>
                      • Your face will be recognized automatically during
                      streams
                    </li>
                    <li>• Non-whitelisted faces will be blurred for privacy</li>
                    <li>• You can manage your whitelist settings anytime</li>
                  </ul>
                </div>
                <div className="flex flex-col sm:flex-row gap-3 justify-center">
                  <Button
                    onClick={() => {
                      setEnrollmentStep("intro");
                      setUserName("");
                      setCapturedImage(null);
                    }}
                    variant="outline"
                    className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                  >
                    Enroll Another Person
                  </Button>
                  <Button className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200">
                    Start Streaming →
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
