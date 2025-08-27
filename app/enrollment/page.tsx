import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CameraCapture } from "@/components/enrollment/CameraCapture";

/**
 * Enrollment Page
 * Whitelist enrollment: camera capture, form, preview.
 * Uses shadcn/ui components.
 */
export default function EnrollmentPage() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-background">
      <Card className="w-full max-w-md shadow-lg">
        <CardHeader>
          <h1 className="text-2xl font-bold">Whitelist Enrollment</h1>
        </CardHeader>
        <CardContent>
          <CameraCapture />
          {/* Enrollment form will go here */}
          <form className="space-y-4">
            <input
              type="text"
              placeholder="Enter your name"
              className="w-full px-3 py-2 border rounded"
            />
            <Button type="submit">Enroll</Button>
          </form>
          {/* FacePreview component will go here */}
          <div className="mt-6">
            <span className="text-muted-foreground">
              Preview will appear here.
            </span>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
