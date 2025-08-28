"use client";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Card } from "@/components/ui/card";

type Face = {
  id: string;
  name: string;
  image: string;
  whitelisted: boolean;
};

interface FacePreviewProps {
  faces: Face[];
}

export function FacePreview({ faces }: FacePreviewProps) {
  return (
    <Card className="p-4 bg-white dark:bg-black border border-black dark:border-white">
      {/* Header */}
      <div className=" border-b border-black dark:border-white">
        <h3 className="font-bold text-base text-black dark:text-white mb-1">
          Face Recognition Preview
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {faces.length} {faces.length === 1 ? "face" : "faces"} detected with
          privacy controls
        </p>
      </div>

      {/* Faces in Horizontal Rows */}
      {faces.length > 0 ? (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {faces.map((face, index) => (
            <div
              key={face.id}
              className="flex items-center gap-3 p-3 border border-black dark:border-white bg-white dark:bg-black hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors"
            >
              {/* Index */}
              <div className="flex-shrink-0 w-6 h-6 bg-black dark:bg-white text-white dark:text-black flex items-center justify-center text-xs font-bold">
                {index + 1}
              </div>

              {/* Avatar */}
              <div className="flex-shrink-0">
                <Avatar className="w-10 h-10 border border-black dark:border-white">
                  <AvatarImage
                    src={face.image}
                    alt={face.name}
                    className={face.whitelisted ? "" : "blur-sm"}
                  />
                  <AvatarFallback className="bg-white dark:bg-black text-black dark:text-white font-bold text-sm">
                    {face.name.charAt(0).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
              </div>

              {/* Name and Status */}
              <div className="flex-grow min-w-0">
                <p className="font-medium text-black dark:text-white text-sm truncate">
                  {face.name}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Privacy: {face.whitelisted ? "OFF" : "ON"}
                </p>
              </div>

              {/* Status Badge */}
              <div className="flex-shrink-0">
                <div
                  className={`px-2 py-1 text-xs font-bold border border-black dark:border-white ${
                    face.whitelisted
                      ? "bg-black dark:bg-white text-white dark:text-black"
                      : "bg-white dark:bg-black text-black dark:text-white"
                  }`}
                >
                  {face.whitelisted ? "VISIBLE" : "BLURRED"}
                </div>
              </div>

              {/* Status Indicator */}
              <div className="flex-shrink-0">
                <div
                  className={`w-3 h-3 ${
                    face.whitelisted
                      ? "bg-black dark:bg-white"
                      : "bg-white dark:bg-black border border-black dark:border-white"
                  }`}
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Empty State */
        <div className="text-center py-8">
          <div className="w-12 h-12 mx-auto mb-3 bg-white dark:bg-black border border-black dark:border-white flex items-center justify-center">
            <div className="w-6 h-6 bg-black dark:bg-white" />
          </div>
          <p className="text-sm font-medium text-black dark:text-white mb-1">
            No Faces Detected
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Faces will appear here when detected by the camera
          </p>
        </div>
      )}
    </Card>
  );
}
